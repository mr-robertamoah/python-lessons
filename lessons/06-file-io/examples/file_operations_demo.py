#!/usr/bin/env python3
"""
File I/O Operations Demo
Demonstrates basic file reading, writing, and CSV processing
"""

import csv
import json
import os
from pathlib import Path

def basic_file_operations():
    """Demonstrate basic file read/write operations"""
    print("=== Basic File Operations ===")
    
    # Writing to a file
    filename = "sample.txt"
    content = """This is a sample file.
It contains multiple lines.
We'll use it to demonstrate file operations."""
    
    with open(filename, 'w') as file:
        file.write(content)
    print(f"Created {filename}")
    
    # Reading entire file
    with open(filename, 'r') as file:
        file_content = file.read()
        print("File contents:")
        print(file_content)
    
    # Reading line by line
    print("\nReading line by line:")
    with open(filename, 'r') as file:
        for line_num, line in enumerate(file, 1):
            print(f"Line {line_num}: {line.strip()}")
    
    # Appending to file
    with open(filename, 'a') as file:
        file.write("\nThis line was appended.")
    
    print(f"\nAppended content to {filename}")
    
    # Clean up
    os.remove(filename)
    print(f"Cleaned up {filename}")

def error_handling_demo():
    """Demonstrate file error handling"""
    print("\n=== Error Handling Demo ===")
    
    # Try to read non-existent file
    try:
        with open("nonexistent.txt", 'r') as file:
            content = file.read()
    except FileNotFoundError:
        print("Handled FileNotFoundError: File doesn't exist")
    
    # Try to write to protected location (may not work on all systems)
    try:
        with open("/root/protected.txt", 'w') as file:
            file.write("This won't work")
    except PermissionError:
        print("Handled PermissionError: No permission to write")
    except OSError as e:
        print(f"Handled OSError: {e}")

def csv_operations_demo():
    """Demonstrate CSV file operations"""
    print("\n=== CSV Operations Demo ===")
    
    # Sample data
    students = [
        ["Name", "Age", "Grade", "Score"],
        ["Alice Johnson", 20, "A", 95],
        ["Bob Smith", 19, "B", 87],
        ["Carol Wilson", 21, "A", 92],
        ["David Brown", 20, "C", 78]
    ]
    
    csv_filename = "students.csv"
    
    # Writing CSV
    with open(csv_filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(students)
    
    print(f"Created {csv_filename}")
    
    # Reading CSV
    print("Reading CSV data:")
    with open(csv_filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            print(f"  {row}")
    
    # Reading CSV with DictReader
    print("\nReading CSV as dictionaries:")
    with open(csv_filename, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            print(f"  {row['Name']}: Grade {row['Grade']}, Score {row['Score']}")
    
    # Calculate average score
    total_score = 0
    count = 0
    with open(csv_filename, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            total_score += int(row['Score'])
            count += 1
    
    average = total_score / count if count > 0 else 0
    print(f"\nClass average score: {average:.1f}")
    
    # Clean up
    os.remove(csv_filename)
    print(f"Cleaned up {csv_filename}")

def json_operations_demo():
    """Demonstrate JSON file operations"""
    print("\n=== JSON Operations Demo ===")
    
    # Sample data
    student_data = {
        "students": [
            {
                "id": 1,
                "name": "Alice Johnson",
                "age": 20,
                "grades": {"math": 95, "science": 92, "english": 88},
                "active": True
            },
            {
                "id": 2,
                "name": "Bob Smith",
                "age": 19,
                "grades": {"math": 87, "science": 84, "english": 90},
                "active": True
            }
        ],
        "class_info": {
            "name": "Data Science 101",
            "semester": "Fall 2024",
            "instructor": "Dr. Smith"
        }
    }
    
    json_filename = "students.json"
    
    # Writing JSON
    with open(json_filename, 'w') as file:
        json.dump(student_data, file, indent=2)
    
    print(f"Created {json_filename}")
    
    # Reading JSON
    with open(json_filename, 'r') as file:
        loaded_data = json.load(file)
    
    print("Loaded JSON data:")
    print(f"Class: {loaded_data['class_info']['name']}")
    print("Students:")
    for student in loaded_data['students']:
        avg_grade = sum(student['grades'].values()) / len(student['grades'])
        print(f"  {student['name']}: Average grade {avg_grade:.1f}")
    
    # Clean up
    os.remove(json_filename)
    print(f"Cleaned up {json_filename}")

def pathlib_demo():
    """Demonstrate pathlib for file path operations"""
    print("\n=== Pathlib Demo ===")
    
    # Create a Path object
    current_dir = Path(".")
    print(f"Current directory: {current_dir.absolute()}")
    
    # Create a file path
    file_path = current_dir / "demo_file.txt"
    print(f"File path: {file_path}")
    
    # Create the file
    file_path.write_text("Hello from pathlib!")
    print(f"Created file: {file_path.name}")
    
    # Read the file
    content = file_path.read_text()
    print(f"File content: {content}")
    
    # File information
    print(f"File exists: {file_path.exists()}")
    print(f"File size: {file_path.stat().st_size} bytes")
    print(f"File suffix: {file_path.suffix}")
    print(f"File stem: {file_path.stem}")
    
    # Clean up
    file_path.unlink()
    print(f"Cleaned up {file_path.name}")

def large_file_processing_demo():
    """Demonstrate processing large files efficiently"""
    print("\n=== Large File Processing Demo ===")
    
    # Create a sample large file
    large_filename = "large_sample.txt"
    
    print("Creating sample large file...")
    with open(large_filename, 'w') as file:
        for i in range(1000):
            file.write(f"This is line {i+1} of the large file.\n")
    
    print(f"Created {large_filename} with 1000 lines")
    
    # Process line by line (memory efficient)
    line_count = 0
    word_count = 0
    
    with open(large_filename, 'r') as file:
        for line in file:
            line_count += 1
            word_count += len(line.split())
            
            # Show progress every 100 lines
            if line_count % 100 == 0:
                print(f"Processed {line_count} lines...")
    
    print(f"File statistics:")
    print(f"  Total lines: {line_count}")
    print(f"  Total words: {word_count}")
    print(f"  Average words per line: {word_count/line_count:.1f}")
    
    # Clean up
    os.remove(large_filename)
    print(f"Cleaned up {large_filename}")

def file_backup_demo():
    """Demonstrate creating file backups"""
    print("\n=== File Backup Demo ===")
    
    import shutil
    from datetime import datetime
    
    # Create original file
    original_file = "important_data.txt"
    with open(original_file, 'w') as file:
        file.write("This is important data that needs backup.\n")
        file.write("Version 1.0\n")
    
    print(f"Created {original_file}")
    
    # Create backup with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = f"backup_{timestamp}_{original_file}"
    
    shutil.copy2(original_file, backup_file)
    print(f"Created backup: {backup_file}")
    
    # Modify original file
    with open(original_file, 'a') as file:
        file.write("Added new data - Version 2.0\n")
    
    print("Modified original file")
    
    # Show difference
    print("\nOriginal file content:")
    with open(original_file, 'r') as file:
        print(file.read())
    
    print("Backup file content:")
    with open(backup_file, 'r') as file:
        print(file.read())
    
    # Clean up
    os.remove(original_file)
    os.remove(backup_file)
    print("Cleaned up files")

def main():
    """Run all demonstrations"""
    print("File I/O Operations Demonstration")
    print("=" * 50)
    
    basic_file_operations()
    error_handling_demo()
    csv_operations_demo()
    json_operations_demo()
    pathlib_demo()
    large_file_processing_demo()
    file_backup_demo()
    
    print("\n" + "=" * 50)
    print("File I/O demonstration complete!")

if __name__ == "__main__":
    main()
