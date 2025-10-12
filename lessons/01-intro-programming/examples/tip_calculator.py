"""
Lesson 01 Example 3: Restaurant Tip Calculator
Demonstrates problem-solving process and basic calculations
"""

print("=== Restaurant Tip Calculator ===")

# Step 1: Define the problem data
bill_amount = 45.50
tip_percentage = 0.18  # 18% tip

# Step 2: Calculate tip and total
tip_amount = bill_amount * tip_percentage
total_amount = bill_amount + tip_amount

# Step 3: Display results
print(f"Bill amount: ${bill_amount}")
print(f"Tip percentage: {tip_percentage * 100}%")
print(f"Tip amount: ${tip_amount:.2f}")
print(f"Total amount: ${total_amount:.2f}")

print("\n=== Multiple Tip Options ===")

# Show different tip percentages
tip_options = [0.15, 0.18, 0.20, 0.25]

for tip_rate in tip_options:
    tip = bill_amount * tip_rate
    total = bill_amount + tip
    print(f"{tip_rate*100}% tip: ${tip:.2f} (Total: ${total:.2f})")
