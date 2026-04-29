"""
chatbot_cli.py
--------------
Command-line interface for the FAQ Chatbot.
Useful for testing without launching the Streamlit UI.

Run with: python chatbot_cli.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.matcher import FAQMatcher
from utils.logger import QueryLogger


def run_cli():
    """Run the FAQ chatbot in the terminal."""

    print("\n" + "="*60)
    print("       FAQ CHATBOT - Command Line Interface")
    print("="*60)
    print("Type your question and press Enter.")
    print("Commands: 'quit' to exit | 'stats' to view log statistics")
    print("="*60 + "\n")

    # Initialize components
    matcher = FAQMatcher(faq_path="data/faqs.json", confidence_threshold=0.2)
    query_logger = QueryLogger(log_dir="logs")

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ('quit', 'exit', 'q'):
                print("\nGoodbye! Have a great day 👋")
                break

            if user_input.lower() == 'stats':
                stats = query_logger.get_stats()
                print("\n--- Query Log Statistics ---")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
                print()
                continue

            # Get answer
            response = matcher.get_answer(user_input)
            query_logger.log(user_input, response)

            print(f"\nBot: {response['answer']}")

            if not response['is_fallback']:
                print(f"     [Matched: \"{response['matched_question']}\" | "
                      f"Confidence: {response['confidence']}% | "
                      f"Category: {response['category']}]")
            else:
                print("     [⚠️  Fallback response — no strong match found]")

            print()

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break


if __name__ == "__main__":
    run_cli()
