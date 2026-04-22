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

        if "tell me about yourself" in category:
            category_guidance = """
You are evaluating a "Tell me about yourself" answer.

What strong answers should do:
- open clearly and professionally
- summarize relevant background
- highlight strengths relevant to the target role
- sound natural and spoken, not like a written essay
- stay concise and not become a life story

What weak answers often do:
- give too much biography
- lack focus
- fail to connect background to the role
- sound vague or generic
"""
        elif "behavioral" in category:
            category_guidance = """
You are evaluating a behavioral interview answer.

What strong answers should do:
- follow a clear STAR-like flow: situation, task, action, result
- emphasize what the candidate personally did
- include enough concrete detail
- end with a result, impact, or lesson learned

What weak answers often do:
- stay too vague
- over-explain background
- fail to show action or ownership
- omit the result
"""
        else:
            category_guidance = """
You are evaluating a general interview answer.

What strong answers should do:
- answer the question directly
- explain reasoning clearly
- connect the answer to the role
- sound confident and concise

What weak answers often do:
- dodge the question
- make claims without support
- sound generic
- include too much filler
"""

        system_prompt = f"""
You are an expert interview coach for English-language job interviews.

Your job is to give high-quality, role-aware coaching that is specific, practical, and encouraging.

{role_context}

{category_guidance}

Scoring dimensions:
- clarity: Is the answer easy to understand?
- structure: Is it organized logically?
- confidence: Does it sound self-assured and professional?
- conciseness: Is it focused without unnecessary detail?

Scoring rules:
- Use the full 1-10 range honestly.
- Do not inflate scores.
- Penalize vague, generic, repetitive, or unfocused answers.
- Reward specificity, relevance, structure, and professional tone.

Feedback rules:
- strengths must be concrete, not generic praise
- tips must be actionable, specific, and realistic
- improvedAnswer must sound like natural spoken English
- improvedAnswer should stay aligned with the question type
- summary should explain the single biggest improvement opportunity clearly
"""

        user_prompt = f"""
Question:
{question}

Candidate answer:
{answer}

Return JSON only with this schema:

{{
  "clarity": integer,
  "structure": integer,
  "confidence": integer,
  "conciseness": integer,
  "strengths": [
    "string",
    "string"
  ],
  "tips": [
    "string",
    "string",
    "string"
  ],
  "improvedAnswer": "string",
  "summary": "string"
}}

Additional requirements:
- strengths must refer to something actually present in the answer
- tips must explain exactly what to improve
- improvedAnswer should be stronger but still realistic for the candidate
- if the category is behavioral, use a STAR-like structure
- if the category is tell me about yourself, keep it concise and role-oriented
- if target role/company is available, use it naturally in your coaching
- do not include markdown
- do not include extra commentary outside the JSON
"""

        response = client.responses.create(
            model="gpt-5.4",
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "answer_feedback",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "clarity": {"type": "integer"},
                            "structure": {"type": "integer"},
                            "confidence": {"type": "integer"},
                            "conciseness": {"type": "integer"},
                            "strengths": {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": 2,
                                "maxItems": 2
                            },
                            "tips": {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": 3,
                                "maxItems": 3
                            },
                            "improvedAnswer": {"type": "string"},
                            "summary": {"type": "string"},
                        },
                        "required": [
                            "clarity",
                            "structure",
                            "confidence",
                            "conciseness",
                            "strengths",
                            "tips",
                            "improvedAnswer",
                            "summary",
                        ],
                        "additionalProperties": False,
                    },
                }
            },
        )

        output_text = response.output_text.strip()
        parsed = json.loads(output_text)

        for key in ["clarity", "structure", "confidence", "conciseness"]:
            try:
                parsed[key] = int(parsed[key])
            except Exception:
                parsed[key] = 6
            parsed[key] = max(1, min(10, parsed[key]))

        if not isinstance(parsed.get("strengths"), list):
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
        parsed["strengths"] = parsed["strengths"][:2]

        if not isinstance(parsed.get("tips"), list):
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
        parsed["tips"] = parsed["tips"][:3]

        if not parsed.get("summary"):
            parsed["summary"] = (
                "Your answer has a good foundation, but it would be stronger with clearer structure "
                "and more specific detail."
            )

        if not parsed.get("improvedAnswer"):
            parsed["improvedAnswer"] = answer

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