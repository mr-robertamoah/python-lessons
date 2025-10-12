"""
Lesson 01 Solutions: Introduction to Programming
Complete solutions for all exercises
"""

print("=== LESSON 01 EXERCISE SOLUTIONS ===\n")

# ============================================================================
# GUIDED EXERCISES
# ============================================================================

print("GUIDED EXERCISE 1: Your First Program")
print("=" * 40)
print("Hello, World!")
print("My name is Python Student")
print("Today is my first day programming!")
print("I'm excited to learn Python!")
print("This is going to be an amazing journey!")
print()

print("GUIDED EXERCISE 2: Python Calculator Practice")
print("=" * 40)
print("25 + 17 =", 25 + 17)
print("100 - 37 =", 100 - 37)
print("12 * 8 =", 12 * 8)
print("144 / 12 =", 144 / 12)
print("3 ** 4 =", 3 ** 4)
print("23 % 7 =", 23 % 7)
print()

print("GUIDED EXERCISE 3: Pizza Party Calculations")
print("=" * 40)
people = 8
pizza_serves = 3
pizza_cost = 12.99

# Calculate number of pizzas needed (round up)
pizzas_needed = (people + pizza_serves - 1) // pizza_serves  # Ceiling division
total_cost = pizzas_needed * pizza_cost
cost_per_person = total_cost / people

print(f"People: {people}")
print(f"Each pizza serves: {pizza_serves}")
print(f"Pizzas needed: {pizzas_needed}")
print(f"Cost per pizza: ${pizza_cost}")
print(f"Total cost: ${total_cost:.2f}")
print(f"Cost per person: ${cost_per_person:.2f}")
print()

# ============================================================================
# INDEPENDENT EXERCISES
# ============================================================================

print("INDEPENDENT EXERCISE 4: Personal Information Display")
print("=" * 40)
print("=== About Me ===")
print("Name: Alex Johnson")
print("Age: 28")
print("City: Seattle")
print("Learning Python because: I want to analyze data at work")
print("Fun fact: I can juggle!")
print()

print("INDEPENDENT EXERCISE 5: Unit Conversions")
print("=" * 40)

# Distance conversion
miles = 50
kilometers = miles * 1.60934
print(f"{miles} miles = {kilometers:.2f} kilometers")

# Weight conversion
pounds = 150
kilograms = pounds * 0.453592
print(f"{pounds} pounds = {kilograms:.2f} kilograms")

# Temperature conversion
fahrenheit = 75
celsius = (fahrenheit - 32) * 5/9
print(f"{fahrenheit}°F = {celsius:.1f}°C")

# Time conversion
hours = 2.5
minutes = hours * 60
seconds = minutes * 60
print(f"{hours} hours = {minutes} minutes = {seconds} seconds")
print()

print("INDEPENDENT EXERCISE 6: Budget Calculator")
print("=" * 40)
monthly_budget = 3000
rent = 1200
food = 400
transportation = 200
entertainment = 150

# Calculate expenses
total_fixed_expenses = rent + food + transportation + entertainment
money_after_fixed = monthly_budget - total_fixed_expenses
savings_amount = money_after_fixed * 0.20
money_for_other = money_after_fixed - savings_amount

print(f"Monthly Budget: ${monthly_budget}")
print(f"Fixed Expenses:")
print(f"  Rent: ${rent}")
print(f"  Food: ${food}")
print(f"  Transportation: ${transportation}")
print(f"  Entertainment: ${entertainment}")
print(f"Total Fixed Expenses: ${total_fixed_expenses}")
print(f"Money after fixed expenses: ${money_after_fixed}")
print(f"Savings (20%): ${savings_amount:.2f}")
print(f"Money left for other expenses: ${money_for_other:.2f}")
print()

print("INDEPENDENT EXERCISE 7: Data Analysis Preview")
print("=" * 40)
monday = 8432
tuesday = 6891
wednesday = 9102
thursday = 7654
friday = 5432
saturday = 12098
sunday = 9876

total_steps = monday + tuesday + wednesday + thursday + friday + saturday + sunday
average_steps = total_steps / 7

print("Daily Steps:")
print(f"Monday: {monday}")
print(f"Tuesday: {tuesday}")
print(f"Wednesday: {wednesday}")
print(f"Thursday: {thursday}")
print(f"Friday: {friday}")
print(f"Saturday: {saturday}")
print(f"Sunday: {sunday}")
print(f"\nTotal steps for the week: {total_steps}")
print(f"Average steps per day: {average_steps:.1f}")
print(f"Highest day: Saturday ({saturday} steps)")
print(f"Lowest day: Friday ({friday} steps)")
print()

# ============================================================================
# CHALLENGE EXERCISES
# ============================================================================

print("CHALLENGE EXERCISE 1: Compound Interest Calculator")
print("=" * 40)

# Scenario 1: $1000 at 5% for 10 years
P1, r1, t1 = 1000, 0.05, 10
A1 = P1 * (1 + r1) ** t1
print(f"${P1} at {r1*100}% for {t1} years = ${A1:.2f}")

# Scenario 2: $5000 at 3% for 20 years
P2, r2, t2 = 5000, 0.03, 20
A2 = P2 * (1 + r2) ** t2
print(f"${P2} at {r2*100}% for {t2} years = ${A2:.2f}")

# Scenario 3: $500 at 7% for 5 years
P3, r3, t3 = 500, 0.07, 5
A3 = P3 * (1 + r3) ** t3
print(f"${P3} at {r3*100}% for {t3} years = ${A3:.2f}")
print()

print("CHALLENGE EXERCISE 2: Grade Calculator")
print("=" * 40)
homework_score = 85
homework_weight = 0.30

midterm_score = 78
midterm_weight = 0.35

final_score = 92
final_weight = 0.35

# Calculate weighted average
weighted_average = (homework_score * homework_weight + 
                   midterm_score * midterm_weight + 
                   final_score * final_weight)

print(f"Homework: {homework_score}% (weight: {homework_weight*100}%)")
print(f"Midterm: {midterm_score}% (weight: {midterm_weight*100}%)")
print(f"Final: {final_score}% (weight: {final_weight*100}%)")
print(f"Weighted Average: {weighted_average:.1f}%")

# Determine letter grade (just the logic, we'll automate this later)
if weighted_average >= 90:
    letter_grade = "A"
elif weighted_average >= 80:
    letter_grade = "B"
elif weighted_average >= 70:
    letter_grade = "C"
elif weighted_average >= 60:
    letter_grade = "D"
else:
    letter_grade = "F"

print(f"Letter Grade: {letter_grade}")
print()

print("CHALLENGE EXERCISE 3: Error Investigation")
print("=" * 40)
print("Common errors and their fixes:")
print()

print("1. print(\"Hello World\"  # Missing closing parenthesis")
print("   Error: SyntaxError: unexpected EOF while parsing")
print("   Fix: print(\"Hello World\")")
print()

print("2. print(hello)  # Undefined variable")
print("   Error: NameError: name 'hello' is not defined")
print("   Fix: print(\"hello\") or define hello = \"some value\" first")
print()

print("3. print(\"5\" + 5)  # Mixing text and numbers")
print("   Error: TypeError: can only concatenate str (not \"int\") to str")
print("   Fix: print(\"5\" + \"5\") or print(5 + 5) or print(\"5\" + str(5))")
print()

print("4. print(10 / 0)  # Division by zero")
print("   Error: ZeroDivisionError: division by zero")
print("   Fix: Check for zero before dividing or use a different denominator")
print()

print("=== END OF SOLUTIONS ===")
