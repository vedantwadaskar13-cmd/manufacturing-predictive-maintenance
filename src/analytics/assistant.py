"""
Precision Manufacturing Plant - AI Manufacturing Assistant Engine
Implements NL-to-SQL intent routing, read-only database execution,
and anti-hallucination structured responses.
"""

import os
import re
import sys
import pandas as pd
from sqlalchemy import create_engine, text

# Database Connection Configuration
DB_USER = os.getenv("MYSQL_USER", "root")
DB_PASSWORD = os.getenv("MYSQL_PASSWORD", "Vw,130505")
DB_HOST = os.getenv("MYSQL_HOST", "localhost")
DB_PORT = os.getenv("MYSQL_PORT", "3306")
DB_NAME = os.getenv("MYSQL_DATABASE", "manufacturing_intel")

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "processed")


class AIManufacturingAssistant:
    def __init__(self, use_mysql: bool = False):
        self.use_mysql = use_mysql
        if self.use_mysql:
            conn_str = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
            self.engine = create_engine(conn_str)
        
        # Load local local offline fallback dataframes
        self.df_risk = pd.read_parquet(os.path.join(PROCESSED_DIR, "current_machine_risk_assessment.parquet"))
        self.df_kpi = pd.read_parquet(os.path.join(PROCESSED_DIR, "kpis", "fleet_kpi_summary.parquet"))
        self.df_diag = pd.read_parquet(os.path.join(PROCESSED_DIR, "machine_shap_diagnostics.parquet"))

    def validate_safe_sql(self, sql_query: str) -> bool:
        """Enforces strict read-only execution guardrails."""
        forbidden_keywords = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE", "GRANT", "CREATE"]
        query_upper = sql_query.upper()
        for kw in forbidden_keywords:
            if re.search(r'\b' + kw + r'\b', query_upper):
                return False
        return True

    def query_database(self, sql_query: str) -> pd.DataFrame:
        """Executes a validated read-only SQL query against MySQL."""
        if not self.validate_safe_sql(sql_query):
            raise ValueError("Security Violation: Non-read-only SQL statement detected!")
        with self.engine.connect() as conn:
            return pd.read_sql(text(sql_query), conn)

    def answer_question(self, user_question: str) -> str:
        """Routes natural language question to validated intent handler."""
        q = user_question.lower().strip()

        # ---------------------------------------------------------
        # Intent 1: Risk & Failure Probability Queries
        # ---------------------------------------------------------
        if any(k in q for k in ["fail", "risk", "critical", "highest probability", "inspect first"]):
            critical_assets = self.df_risk[self.df_risk["composite_risk_score"] >= 55.0]
            if critical_assets.empty:
                return "All machines are currently operating within nominal limits (LOW risk tier)."
            
            resp = ["**Current Machine Risk & Inspection Priority:**\n"]
            for _, r in critical_assets.iterrows():
                resp.append(
                    f"• **{r['machine_id']} ({r['machine_type']})**: "
                    f"Risk Score = **{r['composite_risk_score']}/100** ({r['risk_level']} Tier) | "
                    f"Failure Prob = **{r['failure_probability_pct']}%** | "
                    f"Expected Loss = **₹{r['expected_failure_cost_inr']:,.2f}**\n"
                    f"  - *Prescriptive Action:* {r['prescriptive_action']}\n"
                    f"  - *Primary Drivers:* {r['top_risk_drivers']}\n"
                )
            return "\n".join(resp)

        # ---------------------------------------------------------
        # Intent 2: XAI / Root Cause / "Why" Queries
        # ---------------------------------------------------------
        elif "why" in q or "shap" in q or "root cause" in q or "driver" in q:
            # Extract machine ID if present (e.g., M04, M08)
            match = re.search(r'm\d{2}', q, re.IGNORECASE)
            m_id = match.group(0).upper() if match else "M04"
            
            m_risk = self.df_risk[self.df_risk["machine_id"] == m_id]
            m_diag = self.df_diag[self.df_diag["machine_id"] == m_id]

            if m_risk.empty:
                return f"Machine ID '{m_id}' was not found in the fleet database."

            risk_row = m_risk.iloc[0]
            diag_str = m_diag.iloc[0]["top_root_cause_drivers"] if not m_diag.empty else "N/A"

            return (
                f"**Root-Cause Diagnostic Analysis for {m_id} ({risk_row['machine_type']}):**\n"
                f"• **Composite Risk Score:** {risk_row['composite_risk_score']}/100 ({risk_row['risk_level']} Level)\n"
                f"• **Current Vibration:** {risk_row['vibration_current']} mm/s | **Temperature:** {risk_row['temperature_current']} °C\n"
                f"• **Hours Since Maintenance:** {risk_row['hours_since_maintenance']} hours\n"
                f"• **Top SHAP Predictive Drivers:** {diag_str}\n"
                f"• **Recommended Intervention:** {risk_row['prescriptive_action']}"
            )

        # ---------------------------------------------------------
        # Intent 3: OEE & Production Analytics
        # ---------------------------------------------------------
        elif any(k in q for k in ["oee", "availability", "performance", "quality", "efficiency"]):
            lowest_oee = self.df_kpi.sort_values("oee_pct").iloc[0]
            avg_oee = self.df_kpi["oee_pct"].mean()

            return (
                f"**Fleet OEE Summary:**\n"
                f"• **Fleet Average OEE:** **{avg_oee:.2f}%**\n"
                f"• **Lowest OEE Asset:** **{lowest_oee['machine_id']} ({lowest_oee['machine_type']})** at **{lowest_oee['oee_pct']:.2f}%**\n"
                f"  - *Availability:* {lowest_oee['availability_pct']:.2f}%\n"
                f"  - *Performance:* {lowest_oee['performance_pct']:.2f}%\n"
                f"  - *Quality:* {lowest_oee['quality_pct']:.2f}%\n"
                f"  - *Total Financial Loss:* ₹{lowest_oee['total_cost_of_unreliability_inr']:,.2f}"
            )

        # ---------------------------------------------------------
        # Intent 4: Reliability / MTBF / MTTR Queries
        # ---------------------------------------------------------
        elif any(k in q for k in ["mtbf", "mttr", "downtime", "maintenance cost", "repair"]):
            top_dt = self.df_kpi.sort_values("total_downtime_hours", ascending=False).iloc[0]
            return (
                f"**Fleet Reliability & Downtime Summary:**\n"
                f"• **Fleet Average MTBF:** **{self.df_kpi['mtbf_hours'].mean():.1f} hours**\n"
                f"• **Fleet Average MTTR:** **{self.df_kpi['mttr_hours'].mean():.2f} hours**\n"
                f"• **Most Problematic Asset (Downtime):** **{top_dt['machine_id']}** with **{top_dt['total_downtime_hours']:.1f} hours** of stoppage.\n"
                f"• **Total Fleet Unreliability Cost:** **₹{self.df_kpi['total_cost_of_unreliability_inr'].sum():,.2f}**"
            )

        # ---------------------------------------------------------
        # Default Out-of-Bounds Guardrail Response
        # ---------------------------------------------------------
        else:
            return (
                "I do not have sufficient database context or telemetry parameters to answer that question safely. "
                "You can ask me questions about:\n"
                "  1. Machine risk & failure predictions (e.g., 'Which machines are at risk?')\n"
                "  2. SHAP Root-cause explanations (e.g., 'Why is M04 high risk?')\n"
                "  3. Plant OEE performance (e.g., 'What is the plant OEE?')\n"
                "  4. Reliability metrics (e.g., 'What is our MTBF and MTTR?')"
            )


def main():
    assistant = AIManufacturingAssistant(use_mysql=False)
    
    sample_questions = [
        "Which machines are most likely to fail?",
        "Why is M04 high risk?",
        "What is our plant OEE status?",
        "Which asset has the worst MTBF and highest downtime?"
    ]

    print("=========================================================================================")
    print("               PRECISION MANUFACTURING PLANT - AI ASSISTANT DEMO                         ")
    print("=========================================================================================\n")

    for question in sample_questions:
        print(f"USER QUESTION: {question}")
        print("-" * 80)
        answer = assistant.answer_question(question)
        print(answer)
        print("\n" + "=" * 90 + "\n")


if __name__ == "__main__":
    main()