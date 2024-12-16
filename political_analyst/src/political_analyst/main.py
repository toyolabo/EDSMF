#!/usr/bin/env python
import sys
import warnings

from political_analyst.crew import PoliticalAnalyst

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information


def run():
    """
    Run the crew.
    """
    inputs = {
        "candidate1": "Donald Trump",
        "candidate2": "Kamala Harris",
        "sp500_sectors": "Energy, Industrials, Information Technology, Financials, Materials, Utilities, Health Care, Consumer Discretionary, Consumer Staples, Real Estate, Communication Services",
        "news_reports_path": "../../data/US/news_report_compilation/news_reports.md",
        "dates": "From 2024-10-30 to 2024-11-05",
        "candidate1_pdf_path": "../../data/US/candidate_pdfs/donald_trump.pdf",
        "candidate2_pdf_path": "../../data/US/candidate_pdfs/kamala_harris.pdf",
    }
    PoliticalAnalyst().crew().kickoff(inputs=inputs)


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {"topic": "AI LLMs"}
    try:
        PoliticalAnalyst().crew().train(
            n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs
        )

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")


def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        PoliticalAnalyst().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")


def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {"topic": "AI LLMs"}
    try:
        PoliticalAnalyst().crew().test(
            n_iterations=int(sys.argv[1]), openai_model_name=sys.argv[2], inputs=inputs
        )

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")
