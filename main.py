from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import tempfile
import os
import traceback
import json

app = FastAPI()
client = OpenAI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def healthcheck():
    return {"status": "ok"}

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    temp_path = None
    try:
        suffix = os.path.splitext(file.filename or "")[1]
        if not suffix:
            suffix = ".m4a"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            contents = await file.read()
            temp_file.write(contents)
            temp_path = temp_file.name

        with open(temp_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="gpt-4o-transcribe",
                file=audio_file,
            )

        return {"text": transcription.text}

    except Exception as e:
        print("TRANSCRIPTION ERROR:")
        print(str(e))
        traceback.print_exc()
        return {"error": str(e)}

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/analyze-answer")
async def analyze_answer(payload: dict):
    try:
        answer = payload.get("answer", "").strip()
        question = payload.get("question", "").strip()
        category = payload.get("category", "").strip().lower()
        target_role = payload.get("targetRole", "").strip()
        target_company = payload.get("targetCompany", "").strip()

        if not answer:
            return {"error": "Answer is required."}

        role_context = ""
        if target_role and target_company:
            role_context = f"The candidate is preparing for a {target_role} role at {target_company}."
        elif target_role:
            role_context = f"The candidate is preparing for a {target_role} role."
        elif target_company:
            role_context = f"The candidate is preparing for an interview at {target_company}."

        category_rubric = """
General rubric:
- Clarity: Is the answer easy to understand and direct?
- Structure: Is the answer organized logically?
- Confidence: Does the answer sound assured and professional?
- Conciseness: Is the answer focused without unnecessary detail?
"""

        if "tell me about yourself" in category:
            category_rubric = """
Category-specific rubric: Tell me about yourself
Evaluate the answer on:
- Clarity: Is the self-introduction easy to follow?
- Structure: Does it move logically from background to strengths to fit?
- Confidence: Does the candidate sound professional and self-assured?
- Conciseness: Is it short enough for an interview opening answer?
Special expectations:
- Should not sound like a life story
- Should highlight relevant background
- Should connect the candidate to the role
- Should sound polished and natural in spoken English
"""
        elif "behavioral" in category:
            category_rubric = """
Category-specific rubric: Behavioral interview
Evaluate the answer on:
- Clarity: Is the story easy to follow?
- Structure: Does it follow a strong STAR-like flow (Situation, Task, Action, Result)?
- Confidence: Does the candidate clearly describe what they did?
- Conciseness: Is the story focused without too much irrelevant detail?
Special expectations:
- Should be specific, not vague
- Should emphasize the candidate's actions
- Should include a clear result or lesson
- Should avoid staying too long in background/context
"""
        elif "general" in category:
            category_rubric = """
Category-specific rubric: General interview
Evaluate the answer on:
- Clarity: Is the response direct and understandable?
- Structure: Does it answer the question in a logical order?
- Confidence: Does the candidate sound convincing and intentional?
- Conciseness: Does it stay focused on the main point?
Special expectations:
- Should answer the actual question directly
- Should include reasons, not just claims
- Should sound persuasive and role-relevant
"""

        prompt = f"""
You are a professional interview coach helping a candidate interview in English.

{role_context}

Your task is to evaluate the answer using the scoring rubric below and return ONLY valid JSON.

{category_rubric}

Question:
{question}

Answer:
{answer}

Return ONLY this JSON schema exactly:

{{
  "clarity": integer from 1 to 10,
  "structure": integer from 1 to 10,
  "confidence": integer from 1 to 10,
  "conciseness": integer from 1 to 10,
  "strengths": [
    "one specific strength",
    "one specific strength"
  ],
  "tips": [
    "one specific improvement tip",
    "one specific improvement tip",
    "one specific improvement tip"
  ],
  "improvedAnswer": "a stronger improved version of the candidate's answer in natural spoken English",
  "summary": "2-3 sentence coaching summary explaining the biggest improvement opportunity"
}}

Rules:
- Return JSON only, with no markdown and no extra commentary.
- All scores must be integers from 1 to 10.
- Tips must be specific and actionable, not generic.
- Strengths must mention what the candidate did well.
- Use the target role/company context when relevant.
- The improvedAnswer should sound natural for spoken professional English, not overly formal.
- For behavioral answers, prefer STAR structure in the improvedAnswer.
- For 'Tell me about yourself', keep the improvedAnswer concise and role-oriented.
- The summary should be short, practical, and encouraging.
"""

        response = client.responses.create(
            model="gpt-5.4",
            input=prompt,
        )

        output_text = response.output_text.strip()
        parsed = json.loads(output_text)

        if "clarity" not in parsed:
            parsed["clarity"] = 6
        if "structure" not in parsed:
            parsed["structure"] = 6
        if "confidence" not in parsed:
            parsed["confidence"] = 6
        if "conciseness" not in parsed:
            parsed["conciseness"] = 6

        try:
            parsed["clarity"] = int(parsed["clarity"])
        except Exception:
            parsed["clarity"] = 6
        try:
            parsed["structure"] = int(parsed["structure"])
        except Exception:
            parsed["structure"] = 6
        try:
            parsed["confidence"] = int(parsed["confidence"])
        except Exception:
            parsed["confidence"] = 6
        try:
            parsed["conciseness"] = int(parsed["conciseness"])
        except Exception:
            parsed["conciseness"] = 6

        parsed["clarity"] = max(1, min(10, parsed["clarity"]))
        parsed["structure"] = max(1, min(10, parsed["structure"]))
        parsed["confidence"] = max(1, min(10, parsed["confidence"]))
        parsed["conciseness"] = max(1, min(10, parsed["conciseness"]))

        if "strengths" not in parsed or not isinstance(parsed["strengths"], list):
            parsed["strengths"] = []
        if len(parsed["strengths"]) < 2:
            fallback_strengths = [
                "You addressed the question directly.",
                "Your answer shows relevant motivation and effort.",
            ]
            for item in fallback_strengths:
                if len(parsed["strengths"]) >= 2:
                    break
                parsed["strengths"].append(item)

        if "tips" not in parsed or not isinstance(parsed["tips"], list):
            parsed["tips"] = []
        if len(parsed["tips"]) < 3:
            fallback_tips = [
                "Add more specific detail to strengthen your answer.",
                "Use a clearer structure from beginning to end.",
                "Connect your answer more directly to the role.",
            ]
            for item in fallback_tips:
                if len(parsed["tips"]) >= 3:
                    break
                parsed["tips"].append(item)

        if "summary" not in parsed or not parsed["summary"]:
            parsed["summary"] = (
                "Your answer has a good foundation, but it would be stronger with clearer structure "
                "and more specific detail."
            )

        if "improvedAnswer" not in parsed or not parsed["improvedAnswer"]:
            parsed["improvedAnswer"] = answer

        parsed["strengths"] = parsed["strengths"][:2]
        parsed["tips"] = parsed["tips"][:3]

        return parsed

    except Exception as e:
        print("ANALYZE ERROR:")
        print(str(e))
        traceback.print_exc()
        return {"error": str(e)}

@app.post("/analyze-mock-interview")
async def analyze_mock_interview(payload: dict):
    try:
        category = payload.get("category", "").strip()
        target_role = payload.get("targetRole", "").strip()
        target_company = payload.get("targetCompany", "").strip()
        sessions = payload.get("sessions", [])

        if not sessions or not isinstance(sessions, list):
            return {"error": "Sessions are required."}

        role_context = ""
        if target_role and target_company:
            role_context = f"The candidate is preparing for a {target_role} role at {target_company}."
        elif target_role:
            role_context = f"The candidate is preparing for a {target_role} role."
        elif target_company:
            role_context = f"The candidate is preparing for an interview at {target_company}."

        formatted_sessions = []
        for i, session in enumerate(sessions, start=1):
            formatted_sessions.append(
                f"""
Question Number: {i}
Question:
{session.get("question", "")}

Answer:
{session.get("answer", "")}

Scores:
- Clarity: {session.get("clarity", "")}
- Structure: {session.get("structure", "")}
- Confidence: {session.get("confidence", "")}
- Conciseness: {session.get("conciseness", "")}
"""
            )

        joined_sessions = "\n".join(formatted_sessions)

        prompt = f"""
You are a professional interview coach reviewing a completed mock interview in English.

{role_context}

The candidate completed multiple interview questions in the category:
{category}

Here are the question-by-question results:
{joined_sessions}

Return ONLY valid JSON in this exact schema:

{{
  "overallSummary": "2-4 sentence overall assessment of the mock interview",
  "overallStrengths": [
    "one broad strength pattern",
    "one broad strength pattern"
  ],
  "overallWeaknesses": [
    "one broad weakness pattern",
    "one broad weakness pattern"
  ],
  "nextSteps": [
    "one concrete next-step coaching recommendation",
    "one concrete next-step coaching recommendation",
    "one concrete next-step coaching recommendation"
  ],
  "strongestQuestionNumber": integer,
  "strongestWhy": "1-2 sentence explanation of why this was the strongest answer",
  "weakestQuestionNumber": integer,
  "weakestWhy": "1-2 sentence explanation of why this answer most needs improvement"
}}

Rules:
- Return JSON only.
- No markdown.
- No extra commentary.
- strongestQuestionNumber and weakestQuestionNumber must refer to the Question Number values above.
- strongestQuestionNumber and weakestQuestionNumber should ideally be different if there is more than one question.
- Use the target role/company context when relevant.
- Focus on patterns across the whole mock interview, but still choose one strongest and one weakest answer.
"""

        response = client.responses.create(
            model="gpt-5.4",
            input=prompt,
        )

        output_text = response.output_text.strip()
        parsed = json.loads(output_text)

        if "overallSummary" not in parsed or not parsed["overallSummary"]:
            parsed["overallSummary"] = (
                "This mock interview shows a solid foundation, but the biggest opportunity is to make answers "
                "more consistent in structure, specificity, and role relevance."
            )

        if "overallStrengths" not in parsed or not isinstance(parsed["overallStrengths"], list):
            parsed["overallStrengths"] = []
        if len(parsed["overallStrengths"]) < 2:
            fallback_strengths = [
                "You stayed engaged across multiple questions.",
                "Your answers showed effort and role-relevant intent.",
            ]
            for item in fallback_strengths:
                if len(parsed["overallStrengths"]) >= 2:
                    break
                parsed["overallStrengths"].append(item)

        if "overallWeaknesses" not in parsed or not isinstance(parsed["overallWeaknesses"], list):
            parsed["overallWeaknesses"] = []
        if len(parsed["overallWeaknesses"]) < 2:
            fallback_weaknesses = [
                "Some answers could be more specific and concrete.",
                "Answer structure could be more consistent from start to finish.",
            ]
            for item in fallback_weaknesses:
                if len(parsed["overallWeaknesses"]) >= 2:
                    break
                parsed["overallWeaknesses"].append(item)

        if "nextSteps" not in parsed or not isinstance(parsed["nextSteps"], list):
            parsed["nextSteps"] = []
        if len(parsed["nextSteps"]) < 3:
            fallback_steps = [
                "Practice giving clearer opening sentences.",
                "Use more specific examples to support your answers.",
                "End answers by linking your experience to the role.",
            ]
            for item in fallback_steps:
                if len(parsed["nextSteps"]) >= 3:
                    break
                parsed["nextSteps"].append(item)

        question_count = len(sessions)

        try:
            parsed["strongestQuestionNumber"] = int(parsed.get("strongestQuestionNumber", 1))
        except Exception:
            parsed["strongestQuestionNumber"] = 1

        try:
            parsed["weakestQuestionNumber"] = int(parsed.get("weakestQuestionNumber", question_count))
        except Exception:
            parsed["weakestQuestionNumber"] = question_count

        parsed["strongestQuestionNumber"] = max(1, min(question_count, parsed["strongestQuestionNumber"]))
        parsed["weakestQuestionNumber"] = max(1, min(question_count, parsed["weakestQuestionNumber"]))

        if question_count > 1 and parsed["strongestQuestionNumber"] == parsed["weakestQuestionNumber"]:
            parsed["weakestQuestionNumber"] = 1 if parsed["strongestQuestionNumber"] != 1 else 2

        if "strongestWhy" not in parsed or not parsed["strongestWhy"]:
            parsed["strongestWhy"] = "This answer was the most complete and convincing in the interview."

        if "weakestWhy" not in parsed or not parsed["weakestWhy"]:
            parsed["weakestWhy"] = "This answer has the most room for improvement in structure and specificity."

        parsed["overallStrengths"] = parsed["overallStrengths"][:2]
        parsed["overallWeaknesses"] = parsed["overallWeaknesses"][:2]
        parsed["nextSteps"] = parsed["nextSteps"][:3]

        return parsed

    except Exception as e:
        print("MOCK ANALYZE ERROR:")
        print(str(e))
        traceback.print_exc()
        return {"error": str(e)}