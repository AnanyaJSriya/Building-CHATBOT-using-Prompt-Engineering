"""Basic usage example for Curious Bot."""

from src.curious_bot import CuriousBot

def main():
    # Initialize the bot
    bot = CuriousBot()
    
    # Teach something to Curious
    response = bot.process_input(
        "The Pythagorean theorem states that in a right triangle, "
        "the square of the hypotenuse equals the sum of squares of the other two sides"
    )
    
    print(f"Curious asks: {response}")

if __name__ == "__main__":
    main()
