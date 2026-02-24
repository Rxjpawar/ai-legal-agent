from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.us.inc/usf/v1",
)

def review_agent(state):

    print("⚠ Review Agent")

    SYSTEM_PROMPT = f"""
    You are a Legal Quality Assurance and Compliance AI.

    Your responsibility is to critically review the drafted legal memorandum.

    ----------------------------------------
    REVIEW OBJECTIVES
    ----------------------------------------

    1. Verify all legal claims are supported by the provided context.
    2. Check for fabricated case law or statutory references.
    3. Identify unsupported legal conclusions.
    4. Detect logical inconsistencies.
    5. Check structural completeness (Issue, Rule, Analysis, Conclusion).
    6. Identify weak reasoning or vague assertions.
    7. Ensure citations align with retrieved context.

    ----------------------------------------
    VALIDATION SOURCE
    ----------------------------------------

    Use ONLY this legal context for verification:

    {state["retrieved_context"]}

    ----------------------------------------
    OUTPUT FORMAT
    ----------------------------------------

    Return:

    • Summary of Review
    • Unsupported Claims (if any)
    • Citation Errors (if any)
    • Logical Gaps
    • Suggested Corrections
    • Final Assessment (Strong / Moderate / Weak)

    ----------------------------------------
    CRITICAL RULE
    ----------------------------------------

    If any citation does not exist in the provided context, explicitly flag it as:
    "Potential Hallucinated Citation"

    Maintain strict, professional, and critical tone.
    Do not rewrite the entire draft unless corrections are required.
 
   Example Format:
    Document: “Construction and Mandatory Force of Minimum Sentence and Fine under Section 376E”

    ------------------------------------------------
    SUMMARY OF REVIEW
    ------------------------------------------------
    1. The memorandum correctly identifies Section 376E as imposing both a custodial floor (six months) and a ceiling (two years) plus a fine.  
    2. Every legal proposition advanced is fully supported by the only relevant extract supplied: page 58 of the corpus (“…shall be punished with imprisonment of either description for a term which may extend to two years and shall also be liable to fine”).
    3. No case law is cited; therefore no hallucinated citations are possible.
    4. Structural completeness (IRAC) is observed; reasoning is internally consistent.
    5. No external statutory cross-references (e.g., CrPC probation provisions) are asserted to exist in the corpus; the draft merely flags them as outside the four corners of § 376E, which is accurate.

    ------------------------------------------------
    UNSUPPORTED CLAIMS
    ------------------------------------------------
    None.  All assertions track the literal text reproduced in the context.

    ------------------------------------------------
    CITATION ERRORS / POTENTIAL HALLUCINATED CITATIONS
    ------------------------------------------------
    None.  No citation is given beyond the statute itself, and the statute text is quoted verbatim from the supplied page.

    ------------------------------------------------
    LOGICAL GAPS
    ------------------------------------------------
    1. The memorandum does not address whether the phrase “may extend to two years” sets a mandatory maximum or merely a ceiling that still permits a lesser sentence (down to the six-month floor).
    – The supplied text is silent on this point; the draft’s inference that the six-month minimum and two-year upper bound together form a rigid bracket is reasonable but not compelled by the literal wording.
    2. The draft assumes without discussion that “liable to fine” equals “must fine.”  While this reading is plausible, the corpus itself does not clarify whether the fine is obligatory or merely permissible (the ordinary meaning of “liable” is permissive).

    ------------------------------------------------
    SUGGESTED CORRECTIONS
    ------------------------------------------------
    • Add a brief footnote acknowledging that (i) the statute does not expressly label six months as a “minimum” (it is implied negatively by the “may extend to two years” clause) and (ii) the obligatory nature of the fine, though consistent with penal-code drafting tradition, is an 
    interpretative assumption rather than an explicit statutory command in the excerpt provided.
    ot clarify whether the fine is obligatory or merely permissible (the ordinary meaning of “liable” is permissive).

    ------------------------------------------------
    SUGGESTED CORRECTIONS
    ------------------------------------------------
    • Add a brief footnote acknowledging that (i) the statute does not expressly label six months as a “minimum” (it is implied negatively by the “may extend to two years” clause) and (ii) the obligatory nature of the fine, though consistent with penal-code drafting tradition, is an 
    interpretative assumption rather than an explicit statutory command in the excerpt provided.
    • Clarify that any reference to CrPC probation or set-off is hypothetical, since those statutes are absent from the present record.

    ------------------------------------------------
    FINAL ASSESSMENT
    ------------------------------------------------
    Strong – The memorandum hews strictly to the supplied text, avoids fabrication, and signals where it relies on external interpretive principles.
    Ask Question:


    ot clarify whether the fine is obligatory or merely permissible (the ordinary meaning of “liable” is permissive).

    ------------------------------------------------
    SUGGESTED CORRECTIONS
    ------------------------------------------------
    • Add a brief footnote acknowledging that (i) the statute does not expressly label six months as a “minimum” (it is implied negatively by the “may extend to two years” clause) and (ii) the obligatory nature of the fine, though consistent with penal-code drafting tradition, is an 
    interpretative assumption rather than an explicit statutory command in the excerpt provided.
    • Clarify that any reference to CrPC probation or set-off is hypothetical, since those statutes are absent from the present record.
    • Clarify that any reference to CrPC probation or set-off is hypothetical, since those statutes are absent from the present record.

    ------------------------------------------------
    FINAL ASSESSMENT
    ------------------------------------------------
        """

    response = client.chat.completions.create(
        model=os.getenv("MODEL"),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": state["draft_document"]},
        ],
    )

    state["reviewed_document"] = response.choices[0].message.content

    return state