"""
Lesson 03 Example 3: Number Guessing Game
Demonstrates while loops, conditionals, and user interaction
"""

import random

print("=== Number Guessing Game ===")

# Game setup
secret_number = random.randint(1, 100)
attempts = 0
max_attempts = 7

print("I'm thinking of a number between 1 and 100.")
print(f"You have {max_attempts} attempts to guess it!")

# Main game loop
while attempts < max_attempts:
    try:
        guess = int(input("\nEnter your guess: "))
        attempts += 1
        
        if guess == secret_number:
            print(f"🎉 Congratulations! You guessed it in {attempts} attempts!")
            break
        elif guess < secret_number:
            print("📈 Too low!")
        else:
            print("📉 Too high!")
        
        remaining = max_attempts - attempts
        if remaining > 0:
            print(f"You have {remaining} attempts left.")
        else:
            print(f"💀 Game over! The number was {secret_number}.")
    
    except ValueError:
        print("Please enter a valid number.")
        attempts -= 1  # Don't count invalid input as an attempt
