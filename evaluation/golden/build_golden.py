"""Authoring helper that (re)generates golden_dataset.jsonl.

Questions are written here with *keys* instead of copy-pasted passages: each key
must match exactly one corpus line, and that full line becomes the gold
`reference_contexts` entry. That guarantees gold passages are verbatim corpus
text. The generated JSONL is the version-controlled artifact; tests re-verify it.

    python -m evaluation.golden.build_golden
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent
CORPUS = HERE / "corpus"
OUT = HERE / "golden_dataset.jsonl"
SCHEMA_VERSION = "1"

ABSTAIN_REF = "The documents do not contain this information."


def _lines() -> list[tuple[str, str]]:
    out = []
    for f in sorted(CORPUS.glob("*.md")):
        for raw in f.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line.startswith("- "):
                out.append((f.name, line[2:].strip()))
    return out


def _resolve(key: str) -> tuple[str, str]:
    hits = [(f, l) for f, l in _lines() if key in l]
    if len(hits) != 1:
        raise ValueError(f"key {key!r} matched {len(hits)} corpus lines (needs exactly 1)")
    return hits[0]


# (type, question, reference answer, [keys])
FACTUAL = [
    ("What are the core working hours?", "Core working hours are 10:00 to 16:00 local time, Monday through Friday.", ["Core working hours"]),
    ("What is the policy on carrying over unused annual leave?", "Up to 5 unused annual leave days can be carried over to the next year and must be used by 31 March.", ["Up to 5 unused annual leave days"]),
    ("When is a medical certificate required for sick leave?", "A medical certificate is required after 3 consecutive days of absence.", ["Sick leave is 12 days"]),
    ("How does parental leave differ between caregivers?", "Parental leave is 16 weeks at full pay for the primary caregiver and 6 weeks for the secondary caregiver.", ["Parental leave is 16 weeks"]),
    ("What happens to the data of a deleted workspace?", "Deleted workspaces are permanently purged after 30 days.", ["Deleted workspaces are permanently purged"]),
    ("Which integrations does Lumen Insights support natively?", "Lumen Insights integrates natively with Slack, Salesforce, and BigQuery.", ["integrates natively"]),
    ("What does the Enterprise plan include?", "The Enterprise plan has custom pricing and includes unlimited user seats and single sign-on.", ["Enterprise plan has custom pricing"]),
    ("What does the Slack integration do?", "It can send scheduled report digests at daily or weekly frequency.", ["Slack integration can send"]),
    ("How is data classified under the security policy?", "Data is classified into four levels: Public, Internal, Confidential, and Restricted.", ["four levels"]),
    ("What counts as Restricted data?", "Restricted data includes encryption keys and payment card numbers.", ["Restricted data includes"]),
    ("Where may customer data be processed?", "Customer data may only be processed in data centers located in the European Union.", ["Customer data may only be processed"]),
    ("What compensation do customers get when uptime falls below the SLA?", "Service credits of 10 percent of the monthly fee are issued.", ["Service credits"]),
    ("What are the rules for storing or sending Restricted data?", "Restricted data must never be stored on laptops or sent by email.", ["Restricted data must never"]),
    ("How does the company equity plan work?", "Equity is granted through a stock option plan with a 4-year vesting schedule and a 1-year cliff.", ["stock option plan"]),
]

EXACT = [
    ("How many days of paid annual leave do full-time employees get?", "24 days per calendar year.", ["24 days of paid annual leave"]),
    ("What is the minimum password length?", "At least 14 characters.", ["at least 14 characters"]),
    ("How much is the home office stipend?", "400 euros, paid once after the first 90 days of employment.", ["home office stipend"]),
    ("What is the annual learning budget per employee?", "1,200 euros.", ["learning budget of 1,200"]),
    ("What is the monthly price of the Growth plan?", "199 euros per month.", ["Growth plan costs"]),
    ("How many events per day does the Starter plan allow?", "Up to 10,000 events per day.", ["Starter plan allows up to"]),
    ("What is the maximum size of a single CSV import?", "200 MB.", ["single CSV import"]),
    ("How many widgets can a dashboard contain at most?", "At most 30 widgets.", ["at most 30 widgets"]),
    ("Which encryption standard protects data at rest?", "AES-256.", ["AES-256"]),
    ("What is the recovery time objective?", "4 hours.", ["recovery time objective"]),
    ("How long are database backups retained?", "35 days.", ["backed up every 6 hours"]),
    ("Within how many days must critical vulnerabilities be patched?", "Within 7 days.", ["Critical vulnerabilities must be patched"]),
    ("How long is the probation period?", "6 months.", ["probation period lasts"]),
    ("What is the uptime SLA for the Enterprise plan?", "99.95 percent.", ["uptime SLA"]),
    ("How often does the BigQuery connector sync on the Enterprise plan?", "Every 5 minutes.", ["BigQuery connector"]),
    ("After how many days without login are inactive accounts disabled?", "45 days.", ["Inactive accounts"]),
    ("What is the daily meal reimbursement limit during business travel?", "45 euros per day.", ["Meals during business travel"]),
    ("What is the API rate limit for the Growth plan?", "600 requests per minute.", ["API rate limits are"]),
]

MULTI_HOP = [
    ("How many weeks of parental leave can a couple take in total if one is the primary and the other the secondary caregiver?", "16 + 6 = 22 weeks in total.", ["Parental leave is 16 weeks"]),
    ("If an employee carries over the maximum number of unused days, how many annual leave days can they have in the following year?", "24 days of annual leave plus up to 5 carried-over days is 29 days (carried-over days must be used by 31 March).", ["24 days of paid annual leave", "Up to 5 unused annual leave days"]),
    ("Can an employee work from another country for a whole month?", "No. Working from another country is limited to 20 days per calendar year and HR must be notified at least 30 days in advance.", ["another country"]),
    ("What is the combined value of the home office stipend and the annual learning budget for a new employee in their first year?", "400 euros plus 1,200 euros equals 1,600 euros (the stipend is paid after the first 90 days).", ["home office stipend", "learning budget of 1,200"]),
    ("Can a Starter plan customer use webhooks, and if not which plans are needed?", "No. Webhooks are available on the Growth and Enterprise plans only.", ["Webhooks are available"]),
    ("What would the Growth plan cost per year with the annual billing discount?", "199 euros x 12 = 2,388 euros; with the 15 percent annual discount this is 2,029.80 euros per year.", ["Growth plan costs", "Annual billing gives"]),
    ("How many user seats and how many daily events does the Growth plan include?", "15 user seats and up to 500,000 events per day.", ["Growth plan costs", "Growth plan allows"]),
    ("How quickly must a security incident be reported internally, and how soon must customers be told about a confirmed personal data breach?", "Incidents must be reported within 1 hour of discovery; affected customers must be notified within 72 hours of a confirmed personal data breach.", ["Security incidents must be reported", "Affected customers are notified"]),
    ("How often are databases backed up and what is the recovery point objective?", "Databases are backed up every 6 hours and the recovery point objective is 6 hours.", ["backed up every 6 hours", "recovery time objective"]),
    ("A customer needs 2 years of data history. Which plan can provide that?", "Only the Enterprise plan: Starter keeps 90 days and Growth keeps 13 months, while Enterprise can be configured up to 7 years.", ["Starter plan data is retained", "Growth plan data is retained", "Enterprise customers can configure retention"]),
    ("What must a new hire complete in their first week and how long do they have an onboarding buddy?", "Security awareness training within the first 7 days; the onboarding buddy is assigned for the first 60 days.", ["security awareness training", "onboarding buddy"]),
    ("What is the response target for critical incidents on the Enterprise plan compared with the Growth plan's support response target?", "Enterprise has a 1-hour response target for critical incidents; Growth has an 8 business hour response target.", ["Enterprise customers get a dedicated account manager", "Growth customers get email and chat support"]),
    ("Who must approve production database access and how often are access rights reviewed?", "Production database access needs approval from two people; access rights are reviewed quarterly by the system owner.", ["Production database access", "Access rights are reviewed quarterly"]),
]

AMBIGUOUS = [
    ("How long is the notice period?", "It depends: during probation the notice period is 2 weeks; after probation it is 2 months.", ["notice period during probation", "After probation, the standard notice period"]),
    ("How much leave do I get?", "It depends on the leave type: 24 days annual leave, 12 days sick leave, parental leave of 16 weeks (primary) or 6 weeks (secondary), 2 volunteering days, and bereavement leave of 5 or 2 days.", ["24 days of paid annual leave", "Sick leave is 12 days", "Parental leave is 16 weeks", "2 paid volunteering days", "Bereavement leave is 5 days"]),
    ("What is the limit?", "The question is too vague to answer; the system should ask which limit is meant (events, API rate, CSV size, expenses, etc.).", []),
    ("What is the retention period?", "It depends: product data is kept 90 days (Starter), 13 months (Growth) or up to 7 years (Enterprise); database backups are retained for 35 days.", ["Starter plan data is retained", "Growth plan data is retained", "Enterprise customers can configure retention", "backed up every 6 hours"]),
    ("How fast is the response time?", "It depends on the context: Starter 2 business days, Growth 8 business hours, Enterprise 1 hour for critical incidents; security incidents are acknowledged within 15 minutes.", ["Starter customers get email support", "Growth customers get email and chat support", "Enterprise customers get a dedicated account manager", "acknowledged within 15 minutes"]),
    ("What does the plan cost?", "It depends on the plan: Starter 49 euros per month, Growth 199 euros per month, Enterprise has custom pricing.", ["Starter plan costs", "Growth plan costs", "Enterprise plan has custom pricing"]),
    ("When do reviews happen?", "It depends: performance reviews are in June and December, access rights are reviewed quarterly, and post-incident reviews are completed within 5 business days.", ["Performance reviews take place", "Access rights are reviewed quarterly", "post-incident review"]),
    ("Who needs to approve it?", "The question is too vague; approvals depend on the request (e.g. a manager for expenses above 250 euros, the department head for remote work beyond 3 days per week).", ["Expenses above 250 euros", "more than 3 days per week"]),
]

UNANSWERABLE = [
    "What is the name of the company's CEO?",
    "What is the company's policy on bringing pets to the office?",
    "Does Lumen Insights offer a Snowflake integration?",
    "What is the salary band for a senior engineer?",
    "What was the company's revenue in 2025?",
    "Which programming languages does the Lumen Insights SDK support?",
    "What is the office dress code?",
    "How many employees does Lumen Analytics have?",
    "What is the 401(k) matching percentage?",
    "Does the security policy require SOC 2 certification?",
    "What is the Lumen Insights mobile app rating on the App Store?",
    "What is on the cafeteria menu this week?",
]


def build() -> list[dict]:
    rows: list[dict] = []

    def add(qtype: str, question: str, answer: str, keys: list[str], *, answerable: bool, behavior: str) -> None:
        resolved = [_resolve(k) for k in keys]
        contexts = list(dict.fromkeys(line for _, line in resolved))
        docs = sorted({f for f, _ in resolved})
        rows.append({
            "id": f"q{len(rows) + 1:03d}",
            "question": question,
            "type": qtype,
            "answerable": answerable,
            "expected_behavior": behavior,
            "reference_answer": answer,
            "reference_contexts": contexts,
            "source_documents": docs,
        })

    for q, a, k in FACTUAL:
        add("factual", q, a, k, answerable=True, behavior="answer")
    for q, a, k in EXACT:
        add("exact_match", q, a, k, answerable=True, behavior="answer")
    for q, a, k in MULTI_HOP:
        add("multi_hop", q, a, k, answerable=True, behavior="answer")
    for q, a, k in AMBIGUOUS:
        add("ambiguous", q, a, k, answerable=True, behavior="clarify_or_cover_all")
    for q in UNANSWERABLE:
        add("unanswerable", q, ABSTAIN_REF, [], answerable=False, behavior="abstain")
    return rows


def main() -> None:
    rows = build()
    with OUT.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    by_type: dict[str, int] = {}
    for r in rows:
        by_type[r["type"]] = by_type.get(r["type"], 0) + 1
    print(f"wrote {len(rows)} questions to {OUT} (schema v{SCHEMA_VERSION}): {by_type}")


if __name__ == "__main__":
    main()
