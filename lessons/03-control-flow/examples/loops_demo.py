"""
Lesson 03 Example 2: Loops Demo
Demonstrates for loops, while loops, and loop control
"""

print("=== Loops Demo ===\n")

# Example 1: Basic for loop
print("1. Counting with for loop:")
for i in range(5):
    print(f"Count: {i}")
print()

# Example 2: For loop with range
print("2. Numbers 1 to 5:")
for num in range(1, 6):
    print(f"Number: {num}")
print()

# Example 3: For loop with string
print("3. Letters in 'Python':")
word = "Python"
for letter in word:
    print(f"Letter: {letter}")
print()

# Example 4: Basic while loop
print("4. Countdown with while loop:")
countdown = 5
while countdown > 0:
    print(f"T-minus {countdown}")
    countdown -= 1
print("Blast off!")
print()

# Example 5: Accumulator pattern
print("5. Sum of numbers 1 to 10:")
total = 0
for i in range(1, 11):
    total += i
    print(f"Adding {i}, total now: {total}")
print(f"Final sum: {total}")
print()

# Example 6: Loop with break
print("6. Finding first number divisible by 7:")
for num in range(1, 50):
    if num % 7 == 0:
        print(f"Found it: {num}")
        break
    print(f"Checking {num}... not divisible by 7")
print()

# Example 7: Loop with continue
print("7. Only odd numbers from 1 to 10:")
for num in range(1, 11):
    if num % 2 == 0:  # If even
        continue      # Skip to next iteration
    print(f"Odd number: {num}")
print()

# Example 8: Nested loops
print("8. Simple multiplication table:")
for i in range(1, 4):
    for j in range(1, 4):
        result = i * j
        print(f"{i} x {j} = {result}")
    print()  # Empty line after each row
