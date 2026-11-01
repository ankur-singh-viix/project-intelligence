"""Rule-based fallback brief, built from the enquiry text when live AI analysis fails.

This is deliberately simple keyword matching: it is not AI analysis and says so in
the brief itself. It exists so that a failed provider call never produces output
about an unrelated project (such as the Healthcare sample for a shop).
"""

import re

from core.models import ProjectBrief, TechStack

# Business domains: (category, keyword pattern). The best-scoring domain wins.
_DOMAINS = (
    ("E-commerce Platform", r"e-?commerce|online (?:store|shop)|shopping|\bcart\b|checkout|product|storefront|marketplace|vendor|seller|merchant|catalog|inventory"),
    ("Healthcare Platform", r"health|patient|doctor|clinic|hospital|medical|appointment|prescription|telemedicine"),
    ("Education / E-learning Platform", r"student|teacher|course|lesson|classroom|e-?learning|quiz|school|\blms\b"),
    ("Fintech Application", r"\bbank|wallet|\bloan|lending|fintech|trading|investment|ledger|\bkyc\b"),
    ("Travel & Booking Platform", r"hotel|flight|travel|\btour|reservation|itinerary|\btrip"),
    ("Logistics & Delivery Platform", r"logistics|fleet|delivery|shipment|warehouse|courier|supply chain|\bdriver"),
    ("Project / Workflow Management Tool", r"\btasks?\b|project management|kanban|workflow|ticket|team collaboration"),
    ("Social / Community Platform", r"social network|community|forum|\bfeed\b|follow"),
    ("HR & Recruitment Platform", r"recruit|candidate|hiring|payroll|\bhr\b"),
    ("Real Estate Platform", r"real estate|property|propert|landlord|rental"),
)

# Roles: (pattern, label).
_ROLES = (
    (r"\bcustomers?\b|\bbuyers?\b|\bshoppers?\b", "Customer"),
    (r"\bvendors?\b|\bsellers?\b|\bmerchants?\b", "Vendor"),
    (r"\badmin(?:istrator)?s?\b", "Administrator"),
    (r"\bpatients?\b", "Patient"),
    (r"\bdoctors?\b|\bphysicians?\b", "Doctor"),
    (r"\breceptionists?\b", "Receptionist"),
    (r"\bstudents?\b|\blearners?\b", "Student"),
    (r"\bteachers?\b|\binstructors?\b|\btutors?\b", "Teacher"),
    (r"\bmanagers?\b", "Manager"),
    (r"\bemployees?\b|\bstaff\b", "Staff member"),
    (r"\bdrivers?\b|\bcouriers?\b|\briders?\b", "Driver"),
    (r"\bguests?\b", "Guest"),
)

# Capabilities: (pattern, feature, clarification question, risk, implementation consideration).
_CAPABILITIES = (
    (r"vendor|seller|merchant|marketplace|multi-?vendor", "Vendor registration and storefront management",
     "How are vendors onboarded and approved, and how are commissions or payouts handled?",
     "marketplace payouts, disputes and vendor fraud need clear policies", "model vendors and tenants separately in the data layer"),
    (r"inventory|stock", "Inventory and stock tracking",
     "Is inventory managed per vendor or warehouse, and does it sync with existing systems?",
     "stock counts can drift without transactional updates", "update stock inside the same transaction as order creation"),
    (r"\border", "Order processing and status tracking", None,
     "order state handling (cancellations, refunds, partial fulfilment) adds hidden complexity", "define the order lifecycle before building screens"),
    (r"payment|checkout|\bpay\b|billing|invoice|subscription", "Online payments and checkout",
     "Which payment gateways, currencies and regions must be supported?",
     "payment handling brings PCI-DSS obligations and fraud exposure", "use a hosted payment provider so card data never touches your servers"),
    (r"search|filter|browse|catalog", "Search, filtering and browsing", None,
     "search quality and performance degrade as the catalogue grows", "plan a dedicated search index if the catalogue will be large"),
    (r"deliver|shipping|shipment|track", "Delivery and shipment tracking",
     "Which carriers or logistics partners must be integrated?",
     "carrier integrations depend on third-party API reliability", "isolate carrier integrations behind an adapter layer"),
    (r"register|sign.?up|log.?in|authenticat|account", "User registration and authentication", None,
     "weak authentication or session handling exposes user accounts", "use an established identity library or service rather than custom auth"),
    (r"role-?based|permission|\brbac\b|access control", "Role-based access control", None,
     "incorrect permission rules can expose other users' data", "enforce permissions on the server for every request"),
    (r"notification|e-?mail|\bsms\b|reminder|alert", "Notifications (email, SMS or in-app)",
     "Which notification channels are required, and are there opt-in or regional rules?",
     "notification delivery depends on third-party providers", "send notifications from background jobs"),
    (r"report|analytics|dashboard", "Reporting and dashboards",
     "Which reports and metrics matter most, and who can see them?",
     "heavy reporting queries can slow the main application", "consider a read replica or pre-aggregated tables for reports"),
    (r"chat|messag", "In-app messaging", None,
     "real-time messaging adds scaling and moderation requirements", "evaluate WebSocket support early"),
    (r"\bai\b|artificial intelligence|assistant|chatbot|recommend", "AI assistant or recommendations (scope to be defined)",
     "What should the AI assistant do, and what data may it use?",
     "AI features raise accuracy, privacy and cost questions that are unresolved", "define the AI use case and data sources before choosing a model"),
    (r"appointment|booking|schedul|reservation|availability", "Scheduling and availability management",
     "How are time zones, cancellations and double-booking handled?",
     "scheduling conflicts and time-zone errors", "store times in UTC and enforce availability rules in the database"),
    (r"review|rating", "Ratings and reviews", None,
     "fake or abusive reviews require moderation", "plan moderation tools"),
    (r"upload|document|\bfiles?\b|record", "Document and file management", None,
     "stored files and records need access control and retention rules", "use object storage with signed, expiring URLs"),
    (r"mobile app|\bios\b|android", "Mobile application support",
     "Is a native mobile app required, or is a responsive web app enough?",
     "supporting native mobile apps increases scope and cost", "expose a documented API usable by both web and mobile clients"),
    (r"multi-?tenant|\bsaas\b", "Multi-tenant SaaS architecture",
     "How many tenants are expected, and how must their data be isolated?",
     "tenant data isolation errors are a serious security risk", "decide on the isolation model (shared schema or separate schemas) up front"),
    (r"privacy|secur|gdpr|hipaa|complian|encrypt", "Security and privacy controls",
     "Which regulations or certifications apply (for example GDPR, HIPAA, PCI-DSS)?",
     "regulatory requirements can constrain hosting, storage and logging", "document data flows and retention before building"),
)

_ASSUMED_BASELINE = {
    "E-commerce Platform": ("Product catalogue and listings (assumed)", "Shopping cart (assumed)"),
    "Healthcare Platform": ("Patient and provider records (assumed)",),
    "Education / E-learning Platform": ("Course and lesson management (assumed)",),
    "Fintech Application": ("Account and transaction history (assumed)",),
    "Travel & Booking Platform": ("Search and booking flow (assumed)",),
    "Logistics & Delivery Platform": ("Job assignment and status updates (assumed)",),
    "Project / Workflow Management Tool": ("Task boards and assignment (assumed)",),
    "Social / Community Platform": ("User profiles and content feed (assumed)",),
    "HR & Recruitment Platform": ("Candidate and job tracking (assumed)",),
    "Real Estate Platform": ("Property listings and enquiries (assumed)",),
}

_BASE_QUESTIONS = (
    "Who are the target users, and how many are expected at launch and after a year?",
    "What are the budget and the target launch date?",
    "Which existing systems or third-party services must it integrate with?",
    "Are there hosting, data-residency or compliance requirements?",
    "Should it be a web application, a mobile app, or both?",
)


def _unique(items):
    seen, result = set(), []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _first_sentence(text: str, limit: int = 160) -> str:
    sentence = re.split(r"(?<=[.!?])\s+", " ".join(text.split()), maxsplit=1)[0]
    return sentence if len(sentence) <= limit else sentence[: limit - 3].rstrip() + "..."


def _category(lowered: str) -> str:
    scores = [(len(set(re.findall(pattern, lowered))), name) for name, pattern in _DOMAINS]
    best_score, best = max(scores, key=lambda s: s[0])
    if best_score == 0:
        category = "General Software Project"
    else:
        category = best
        if best == "E-commerce Platform" and re.search(r"vendor|seller|merchant|multi-?vendor", lowered):
            category = "Multi-vendor E-commerce Marketplace"
    if re.search(r"\bsaas\b", lowered) and "SaaS" not in category:
        category += " (SaaS)"
    return category


def build_rule_based_brief(enquiry_text: str) -> ProjectBrief:
    """Build a ProjectBrief from the enquiry using keyword rules (no AI involved).

    Every statement is derived from words in the enquiry or marked "(assumed)".
    The brief states in its assumptions that it is rule-based.
    """
    text = " ".join(str(enquiry_text).split())
    lowered = text.lower()
    category = _category(lowered)

    matched = [cap for cap in _CAPABILITIES if re.search(cap[0], lowered)]
    roles = [label for pattern, label in _ROLES if re.search(pattern, lowered)]
    if not roles and re.search(r"\busers?\b", lowered):
        roles = ["User"]

    features = []
    if text:
        features.append(f"Business objective: {_first_sentence(text)}")
    features += [
        f"User role: {role} - mentioned in the enquiry; permissions to be confirmed" for role in roles
    ]
    features += [cap[1] for cap in matched]
    features += list(_ASSUMED_BASELINE.get(category.replace(" (SaaS)", ""), ()))
    if not matched:
        features.append("Core functionality to be defined with the client (assumed)")

    backend = "Python / FastAPI (suggested default)"
    if re.search(r"\bchat|messag|real-?time|\blive\b", lowered):
        backend = "Python / FastAPI with WebSocket support (suggested default)"
    stack = TechStack(
        frontend="React (suggested default)",
        backend=backend,
        database="PostgreSQL (suggested default)",
    )

    questions = _unique([cap[2] for cap in matched] + list(_BASE_QUESTIONS))[:8]

    risks = _unique(
        [cap[3] for cap in matched]
        + [
            "the enquiry leaves scope, scale and integrations open, so effort estimates are uncertain",
            "security and access control must be designed from the start",
        ]
    )
    assumptions = [
        "this brief was generated by simple keyword rules from the enquiry text, not by an AI model",
        "it is a web application unless a mobile app is requested (assumed)",
        "the suggested technology is a default, not a client requirement (assumed)",
    ]
    implementation = _unique(
        [cap[4] for cap in matched]
        + ["confirm the open questions with the client before estimating"]
    )
    risk_text = (
        f"Risks: {'; '.join(risks)}. "
        f"Assumptions: {'; '.join(assumptions)}. "
        f"Implementation considerations: {'; '.join(implementation)}."
    )

    return ProjectBrief(
        project_category=category,
        core_features=features,
        recommended_tech_stack=stack,
        missing_information_questions=questions,
        risk_assessment=risk_text,
    )
