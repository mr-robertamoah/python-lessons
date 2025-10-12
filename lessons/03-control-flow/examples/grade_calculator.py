"""
Lesson 03 Example 3: Grade Calculator
Practical example combining conditionals and user input
"""

print("=== Grade Calculator ===")

# Get student information
student_name = input("Enter student name: ")
print(f"\nCalculating grades for {student_name}")

# Get scores with validation
while True:
    try:
        homework = float(input("Homework average (0-100): "))
        if 0 <= homework <= 100:
            break
        else:
            print("Please enter a score between 0 and 100.")
    except ValueError:
        print("Please enter a valid number.")

while True:
    try:
        midterm = float(input("Midterm score (0-100): "))
        if 0 <= midterm <= 100:
            break
        else:
            print("Please enter a score between 0 and 100.")
    except ValueError:
        print("Please enter a valid number.")

while True:
    try:
        final = float(input("Final exam score (0-100): "))
        if 0 <= final <= 100:
            break
        else:
            print("Please enter a score between 0 and 100.")
    except ValueError:
        print("Please enter a valid number.")

# Calculate weighted average (30% homework, 30% midterm, 40% final)
total_score = (homework * 0.3) + (midterm * 0.3) + (final * 0.4)

# Determine letter grade and message
if total_score >= 97:
    letter = "A+"
    message = "Outstanding work!"
elif total_score >= 93:
    letter = "A"
    message = "Excellent work!"
elif total_score >= 90:
    letter = "A-"
    message = "Great job!"
elif total_score >= 87:
    letter = "B+"
    message = "Very good work!"
elif total_score >= 83:
    letter = "B"
    message = "Good job!"
elif total_score >= 80:
    letter = "B-"
    message = "Solid work!"
elif total_score >= 77:
    letter = "C+"
    message = "Satisfactory work."
elif total_score >= 73:
    letter = "C"
    message = "Acceptable work."
elif total_score >= 70:
    letter = "C-"
    message = "Minimum passing grade."
elif total_score >= 67:
    letter = "D+"
    message = "Below average. Needs improvement."
elif total_score >= 65:
    letter = "D"
    message = "Poor performance. Significant improvement needed."
else:
    letter = "F"
    message = "Failing grade. Please see instructor immediately."

# Display results
print(f"\n=== Grade Report for {student_name} ===")
print(f"Homework Average: {homework:.1f}% (30% weight)")
print(f"Midterm Score: {midterm:.1f}% (30% weight)")
print(f"Final Exam Score: {final:.1f}% (40% weight)")
print(f"Overall Score: {total_score:.1f}%")
print(f"Letter Grade: {letter}")
print(f"Comment: {message}")

# Additional feedback based on individual scores
print(f"\n=== Individual Score Analysis ===")
if homework < 70:
    print("⚠️  Homework average is below 70%. Consider improving study habits.")
if midterm < 70:
    print("⚠️  Midterm score is below 70%. Review course material.")
if final < 70:
    print("⚠️  Final exam score is below 70%. Consider additional study time.")

# Check for improvement trend
if final > midterm and midterm > homework:
    print("📈 Great improvement trend! Scores are getting better.")
elif final < midterm and midterm < homework:
    print("📉 Declining trend. Consider seeking additional help.")
else:
    print("📊 Mixed performance. Focus on consistency.")
