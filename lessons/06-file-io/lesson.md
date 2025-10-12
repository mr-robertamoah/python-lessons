# Lesson 06: File Input/Output - Working with Files and Data

## Learning Objectives
By the end of this lesson, you will be able to:
- Read and write text files in Python
- Work with CSV files for structured data
- Handle file-related errors gracefully
- Use context managers for proper file handling
- Process real-world data files
- Understand different file formats and encodings

## Why File I/O Matters for Data Analysis

### Real-World Data Sources
Most data analysis starts with reading data from files:
- **CSV files**: Spreadsheet data, survey results, sales records
- **Text files**: Log files, configuration files, reports
- **JSON files**: Web API responses, configuration data
- **Excel files**: Business reports, financial data

```python
# Instead of hardcoding data like this:
sales_data = [1000, 1200, 950, 1100, 1300]

# You'll read it from files like this:
with open('sales_data.csv', 'r') as file:
    sales_data = [float(line.strip()) for line in file]
```

## Basic File Operations

### Opening and Closing Files
```python
# Basic file opening (not recommended)
file = open('data.txt', 'r')
content = file.read()
file.close()  # Must remember to close!

# Better: Using context manager (recommended)
with open('data.txt', 'r') as file:
    content = file.read()
# File automatically closes when leaving the 'with' block
```

### File Modes
```python
# Read modes
with open('file.txt', 'r') as f:    # Read text (default)
    content = f.read()

with open('file.txt', 'rb') as f:   # Read binary
    content = f.read()

# Write modes
with open('file.txt', 'w') as f:    # Write text (overwrites existing)
    f.write("New content")

with open('file.txt', 'a') as f:    # Append text
    f.write("Additional content")

# Read and write
with open('file.txt', 'r+') as f:   # Read and write
    content = f.read()
    f.write("More content")
```

## Reading Files

### Reading Entire File
```python
# Read all content as one string
with open('story.txt', 'r') as file:
    content = file.read()
    print(content)

# Read all lines into a list
with open('story.txt', 'r') as file:
    lines = file.readlines()
    for line in lines:
        print(line.strip())  # strip() removes newline characters
```

### Reading Line by Line
```python
# Memory-efficient for large files
with open('large_file.txt', 'r') as file:
    for line in file:
        # Process one line at a time
        processed_line = line.strip().upper()
        print(processed_line)

# Read specific number of lines
with open('data.txt', 'r') as file:
    first_line = file.readline()
    second_line = file.readline()
    print(f"First: {first_line.strip()}")
    print(f"Second: {second_line.strip()}")
```

### Practical Reading Example
```python
def analyze_text_file(filename):
    """Analyze a text file and return statistics"""
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()
        
        # Calculate statistics
        total_lines = len(lines)
        total_words = 0
        total_chars = 0
        
        for line in lines:
            words = line.split()
            total_words += len(words)
            total_chars += len(line)
        
        return {
            "lines": total_lines,
            "words": total_words,
            "characters": total_chars,
            "avg_words_per_line": round(total_words / total_lines, 2) if total_lines > 0 else 0
        }
    
    except FileNotFoundError:
        return {"error": "File not found"}
    except Exception as e:
        return {"error": str(e)}

# Usage
stats = analyze_text_file("sample.txt")
print(stats)
```

## Writing Files

### Basic Writing
```python
# Write string to file
data = "Hello, World!\nThis is a new line."
with open('output.txt', 'w') as file:
    file.write(data)

# Write multiple lines
lines = ["First line", "Second line", "Third line"]
with open('output.txt', 'w') as file:
    for line in lines:
        file.write(line + '\n')

# Or use writelines (but add newlines yourself)
with open('output.txt', 'w') as file:
    file.writelines([line + '\n' for line in lines])
```

### Appending to Files
```python
# Add to existing file
with open('log.txt', 'a') as file:
    file.write(f"New log entry at {datetime.now()}\n")

# Append multiple entries
log_entries = [
    "User logged in",
    "User viewed dashboard", 
    "User logged out"
]

with open('activity.log', 'a') as file:
    for entry in log_entries:
        file.write(f"{entry}\n")
```

### Writing Structured Data
```python
def save_student_grades(students, filename):
    """Save student data to a text file"""
    with open(filename, 'w') as file:
        file.write("Student Grade Report\n")
        file.write("=" * 30 + "\n")
        
        for student in students:
            name = student["name"]
            grades = student["grades"]
            average = sum(grades) / len(grades)
            
            file.write(f"Student: {name}\n")
            file.write(f"Grades: {', '.join(map(str, grades))}\n")
            file.write(f"Average: {average:.2f}\n")
            file.write("-" * 20 + "\n")

# Example usage
students = [
    {"name": "Alice", "grades": [85, 92, 78, 96]},
    {"name": "Bob", "grades": [88, 76, 91, 83]}
]

save_student_grades(students, "grade_report.txt")
```

## Working with CSV Files

### Reading CSV Files
```python
import csv

# Basic CSV reading
def read_csv_basic(filename):
    """Read CSV file manually"""
    data = []
    with open(filename, 'r') as file:
        for line in file:
            # Split by comma and strip whitespace
            row = [item.strip() for item in line.split(',')]
            data.append(row)
    return data

# Using csv module (recommended)
def read_csv_proper(filename):
    """Read CSV using csv module"""
    data = []
    with open(filename, 'r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(row)
    return data

# Reading with headers
def read_csv_with_headers(filename):
    """Read CSV and return as list of dictionaries"""
    data = []
    with open(filename, 'r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(dict(row))
    return data
```

### Writing CSV Files
```python
import csv

def write_csv_basic(data, filename):
    """Write data to CSV file"""
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        for row in data:
            writer.writerow(row)

def write_csv_with_headers(data, headers, filename):
    """Write data with headers to CSV file"""
    with open(filename, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        for row in data:
            writer.writerow(row)

# Example: Save sales data
sales_data = [
    {"date": "2024-01-01", "product": "Widget A", "sales": 100, "revenue": 1000},
    {"date": "2024-01-02", "product": "Widget B", "sales": 150, "revenue": 1800},
    {"date": "2024-01-03", "product": "Widget A", "sales": 120, "revenue": 1200}
]

headers = ["date", "product", "sales", "revenue"]
write_csv_with_headers(sales_data, headers, "sales_report.csv")
```

### CSV Data Analysis Example
```python
import csv

def analyze_sales_csv(filename):
    """Analyze sales data from CSV file"""
    try:
        with open(filename, 'r', newline='') as file:
            reader = csv.DictReader(file)
            
            total_sales = 0
            total_revenue = 0
            product_sales = {}
            
            for row in reader:
                # Convert string numbers to integers/floats
                sales = int(row['sales'])
                revenue = float(row['revenue'])
                product = row['product']
                
                total_sales += sales
                total_revenue += revenue
                
                # Track sales by product
                if product in product_sales:
                    product_sales[product] += sales
                else:
                    product_sales[product] = sales
            
            # Find best-selling product
            best_product = max(product_sales, key=product_sales.get)
            
            return {
                "total_sales": total_sales,
                "total_revenue": total_revenue,
                "average_revenue_per_sale": round(total_revenue / total_sales, 2),
                "product_sales": product_sales,
                "best_selling_product": best_product
            }
    
    except FileNotFoundError:
        return {"error": "CSV file not found"}
    except Exception as e:
        return {"error": f"Error processing CSV: {str(e)}"}

# Usage
analysis = analyze_sales_csv("sales_data.csv")
print("Sales Analysis:")
for key, value in analysis.items():
    print(f"  {key.replace('_', ' ').title()}: {value}")
```

## Exception Handling

### Common File Errors
```python
def safe_file_read(filename):
    """Safely read a file with proper error handling"""
    try:
        with open(filename, 'r') as file:
            return file.read()
    
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
        return None
    
    except PermissionError:
        print(f"Error: Permission denied to read '{filename}'")
        return None
    
    except UnicodeDecodeError:
        print(f"Error: Cannot decode '{filename}' - try different encoding")
        return None
    
    except Exception as e:
        print(f"Unexpected error reading '{filename}': {str(e)}")
        return None

# Usage
content = safe_file_read("data.txt")
if content is not None:
    print("File read successfully!")
    print(content[:100])  # Print first 100 characters
```

### Robust File Processing
```python
def process_data_file(input_file, output_file):
    """Process data file with comprehensive error handling"""
    try:
        # Read input file
        with open(input_file, 'r') as infile:
            lines = infile.readlines()
        
        # Process data
        processed_lines = []
        for i, line in enumerate(lines, 1):
            try:
                # Example: convert to uppercase and add line number
                processed = f"{i:03d}: {line.strip().upper()}\n"
                processed_lines.append(processed)
            except Exception as e:
                print(f"Warning: Error processing line {i}: {e}")
                continue
        
        # Write output file
        with open(output_file, 'w') as outfile:
            outfile.writelines(processed_lines)
        
        return f"Successfully processed {len(processed_lines)} lines"
    
    except FileNotFoundError:
        return f"Error: Input file '{input_file}' not found"
    
    except PermissionError as e:
        return f"Error: Permission denied - {e}"
    
    except Exception as e:
        return f"Unexpected error: {e}"

# Usage
result = process_data_file("input.txt", "output.txt")
print(result)
```

## Working with Different File Formats

### JSON Files
```python
import json

def read_json_file(filename):
    """Read JSON data from file"""
    try:
        with open(filename, 'r') as file:
            data = json.load(file)
        return data
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {filename}: {e}")
        return None
    except FileNotFoundError:
        print(f"Error: File {filename} not found")
        return None

def write_json_file(data, filename):
    """Write data to JSON file"""
    try:
        with open(filename, 'w') as file:
            json.dump(data, file, indent=2)
        return True
    except Exception as e:
        print(f"Error writing JSON file: {e}")
        return False

# Example: Configuration file
config = {
    "database": {
        "host": "localhost",
        "port": 5432,
        "name": "myapp"
    },
    "settings": {
        "debug": True,
        "max_connections": 100
    }
}

write_json_file(config, "config.json")
loaded_config = read_json_file("config.json")
```

### File Encoding
```python
def read_file_with_encoding(filename, encoding='utf-8'):
    """Read file with specific encoding"""
    try:
        with open(filename, 'r', encoding=encoding) as file:
            return file.read()
    except UnicodeDecodeError:
        # Try different encodings
        encodings = ['latin-1', 'cp1252', 'iso-8859-1']
        for enc in encodings:
            try:
                with open(filename, 'r', encoding=enc) as file:
                    print(f"Successfully read with {enc} encoding")
                    return file.read()
            except UnicodeDecodeError:
                continue
        
        print("Could not decode file with any common encoding")
        return None
```

## Practical Data Processing Examples

### Example 1: Log File Analysis
```python
import re
from datetime import datetime

def analyze_log_file(filename):
    """Analyze web server log file"""
    ip_counts = {}
    status_counts = {}
    total_requests = 0
    
    try:
        with open(filename, 'r') as file:
            for line in file:
                # Simple log format: IP - - [timestamp] "request" status size
                parts = line.split()
                if len(parts) >= 9:
                    ip = parts[0]
                    status = parts[8]
                    
                    # Count IPs
                    ip_counts[ip] = ip_counts.get(ip, 0) + 1
                    
                    # Count status codes
                    status_counts[status] = status_counts.get(status, 0) + 1
                    
                    total_requests += 1
        
        # Find top IPs
        top_ips = sorted(ip_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "total_requests": total_requests,
            "unique_ips": len(ip_counts),
            "top_ips": top_ips,
            "status_codes": status_counts
        }
    
    except Exception as e:
        return {"error": str(e)}

# Usage
log_analysis = analyze_log_file("access.log")
print("Log Analysis:")
for key, value in log_analysis.items():
    print(f"  {key}: {value}")
```

### Example 2: Survey Data Processing
```python
def process_survey_responses(input_csv, output_csv):
    """Process survey responses and generate summary"""
    import csv
    
    responses = []
    
    # Read survey data
    try:
        with open(input_csv, 'r', newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                # Clean and validate data
                try:
                    response = {
                        "age": int(row["age"]),
                        "satisfaction": int(row["satisfaction"]),
                        "department": row["department"].strip().title(),
                        "years_employed": int(row["years_employed"])
                    }
                    
                    # Validate ranges
                    if 18 <= response["age"] <= 100 and 1 <= response["satisfaction"] <= 10:
                        responses.append(response)
                
                except (ValueError, KeyError) as e:
                    print(f"Skipping invalid row: {e}")
                    continue
        
        # Calculate statistics
        if responses:
            avg_age = sum(r["age"] for r in responses) / len(responses)
            avg_satisfaction = sum(r["satisfaction"] for r in responses) / len(responses)
            
            # Department breakdown
            dept_stats = {}
            for response in responses:
                dept = response["department"]
                if dept not in dept_stats:
                    dept_stats[dept] = {"count": 0, "satisfaction_sum": 0}
                
                dept_stats[dept]["count"] += 1
                dept_stats[dept]["satisfaction_sum"] += response["satisfaction"]
            
            # Calculate department averages
            for dept in dept_stats:
                count = dept_stats[dept]["count"]
                total_sat = dept_stats[dept]["satisfaction_sum"]
                dept_stats[dept]["avg_satisfaction"] = round(total_sat / count, 2)
            
            # Write summary
            with open(output_csv, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Survey Summary"])
                writer.writerow(["Total Responses", len(responses)])
                writer.writerow(["Average Age", round(avg_age, 1)])
                writer.writerow(["Average Satisfaction", round(avg_satisfaction, 2)])
                writer.writerow([])
                writer.writerow(["Department", "Count", "Avg Satisfaction"])
                
                for dept, stats in dept_stats.items():
                    writer.writerow([dept, stats["count"], stats["avg_satisfaction"]])
            
            return f"Processed {len(responses)} responses successfully"
        
        else:
            return "No valid responses found"
    
    except Exception as e:
        return f"Error processing survey: {e}"

# Usage
result = process_survey_responses("survey_raw.csv", "survey_summary.csv")
print(result)
```

## Best Practices

### 1. Always Use Context Managers
```python
# Good: Automatic file closing
with open('file.txt', 'r') as f:
    content = f.read()

# Avoid: Manual file handling
f = open('file.txt', 'r')
content = f.read()
f.close()  # Easy to forget!
```

### 2. Handle Errors Gracefully
```python
def safe_file_operation(filename):
    try:
        with open(filename, 'r') as file:
            return file.read()
    except FileNotFoundError:
        print(f"File {filename} not found")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None
```

### 3. Use Appropriate File Modes
```python
# Reading existing file
with open('data.txt', 'r') as f:
    content = f.read()

# Creating new file (overwrites existing)
with open('output.txt', 'w') as f:
    f.write("New content")

# Adding to existing file
with open('log.txt', 'a') as f:
    f.write("New log entry\n")
```

### 4. Process Large Files Efficiently
```python
# Good: Process line by line for large files
def process_large_file(filename):
    with open(filename, 'r') as file:
        for line in file:  # Memory efficient
            process_line(line)

# Avoid: Loading entire large file into memory
def process_large_file_bad(filename):
    with open(filename, 'r') as file:
        all_lines = file.readlines()  # Could use too much memory
        for line in all_lines:
            process_line(line)
```

## Key Terminology

- **File Handle**: Object representing an open file
- **Context Manager**: Object that defines runtime context (like `with` statement)
- **File Mode**: How a file is opened (read, write, append, etc.)
- **Encoding**: How text is converted to/from bytes (UTF-8, ASCII, etc.)
- **CSV**: Comma-Separated Values file format
- **JSON**: JavaScript Object Notation, text-based data format
- **Exception**: Error that occurs during program execution
- **Buffer**: Temporary storage area for file data

## Looking Ahead

In Lesson 07, we'll learn about:
- **Testing**: Writing tests to verify code works correctly
- **Debugging**: Finding and fixing errors in code
- **Unit tests**: Testing individual functions
- **Test-driven development**: Writing tests before code
