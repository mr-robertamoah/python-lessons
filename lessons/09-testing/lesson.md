# Lesson 07: Testing and Debugging - Writing Reliable Code

## Learning Objectives
By the end of this lesson, you will be able to:
- Debug Python code using various techniques
- Write unit tests to verify code correctness
- Use Python's unittest and pytest frameworks
- Apply test-driven development principles
- Handle and prevent common programming errors
- Use debugging tools and techniques effectively

## Why Testing and Debugging Matter

### Real-World Consequences
In data analysis, bugs can lead to:
- **Incorrect conclusions**: Wrong business decisions based on faulty analysis
- **Lost time**: Hours spent tracking down errors in complex analyses
- **Lost trust**: Stakeholders losing confidence in your results
- **Financial impact**: Wrong data leading to costly mistakes

```python
# A small bug with big consequences
def calculate_profit_margin(revenue, costs):
    # Bug: Using + instead of -
    profit = revenue + costs  # Should be revenue - costs
    return (profit / revenue) * 100

# This would show positive margins even when losing money!
margin = calculate_profit_margin(1000, 1200)  # Should be -20%, shows 220%
```

### Benefits of Testing
- **Confidence**: Know your code works correctly
- **Documentation**: Tests show how code should be used
- **Refactoring safety**: Change code without breaking functionality
- **Bug prevention**: Catch errors before they cause problems

## Debugging Techniques

### 1. Print Debugging
The simplest debugging technique - add print statements to see what's happening:

```python
def calculate_average(numbers):
    print(f"Input numbers: {numbers}")  # Debug: see input
    
    if not numbers:
        print("Empty list detected")  # Debug: check condition
        return 0
    
    total = sum(numbers)
    print(f"Sum: {total}")  # Debug: check calculation
    
    count = len(numbers)
    print(f"Count: {count}")  # Debug: check count
    
    average = total / count
    print(f"Average: {average}")  # Debug: see result
    
    return average

# Test with debug output
result = calculate_average([10, 20, 30])
```

### 2. Using the Debugger
Python's built-in debugger (pdb) allows you to step through code:

```python
import pdb

def problematic_function(data):
    pdb.set_trace()  # Execution will pause here
    
    processed = []
    for item in data:
        # You can examine variables here
        result = item * 2
        processed.append(result)
    
    return processed

# When you run this, you'll get an interactive debugger
# Commands: n (next), s (step), c (continue), p variable_name (print)
```

### 3. Logging Instead of Print
For production code, use logging instead of print statements:

```python
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

def analyze_sales_data(sales_data):
    logging.info(f"Starting analysis of {len(sales_data)} records")
    
    try:
        total_sales = sum(sale['amount'] for sale in sales_data)
        logging.debug(f"Total sales calculated: {total_sales}")
        
        average_sale = total_sales / len(sales_data)
        logging.debug(f"Average sale: {average_sale}")
        
        return {
            'total': total_sales,
            'average': average_sale,
            'count': len(sales_data)
        }
    
    except Exception as e:
        logging.error(f"Error in analysis: {e}")
        raise

# Usage
sales = [{'amount': 100}, {'amount': 200}, {'amount': 150}]
result = analyze_sales_data(sales)
```

### 4. Common Debugging Strategies

#### Rubber Duck Debugging
Explain your code line by line to someone (or a rubber duck). Often you'll spot the error while explaining.

#### Binary Search Debugging
Comment out half your code to isolate where the problem occurs:

```python
def complex_calculation(data):
    # Step 1: Data validation
    # if not data:
    #     return None
    
    # Step 2: Initial processing
    processed = [x * 2 for x in data]
    
    # Step 3: Complex calculation
    # result = sum(processed) / len(processed)
    
    # Step 4: Final formatting
    # return round(result, 2)
    
    return processed  # Test just this part first
```

#### Minimal Reproducible Example
Create the smallest possible code that demonstrates the problem:

```python
# Instead of debugging a 100-line function,
# create a minimal version that shows the issue

# Original complex function
def complex_data_processor(data, options, filters, transformations):
    # 100 lines of code with a bug somewhere...
    pass

# Minimal reproduction
def simple_test():
    data = [1, 2, 3]
    result = data[5]  # IndexError - found the issue!
    return result
```

## Introduction to Unit Testing

### What are Unit Tests?
Unit tests are small pieces of code that test individual functions or methods to ensure they work correctly.

```python
# Function to test
def add_numbers(a, b):
    return a + b

# Manual testing (what we've been doing)
print(add_numbers(2, 3))  # Should be 5
print(add_numbers(-1, 1))  # Should be 0

# Unit test (automated testing)
def test_add_numbers():
    assert add_numbers(2, 3) == 5
    assert add_numbers(-1, 1) == 0
    assert add_numbers(0, 0) == 0
    print("All tests passed!")

test_add_numbers()
```

### Using Python's unittest Module

```python
import unittest

class TestMathFunctions(unittest.TestCase):
    
    def test_add_positive_numbers(self):
        """Test adding positive numbers"""
        result = add_numbers(2, 3)
        self.assertEqual(result, 5)
    
    def test_add_negative_numbers(self):
        """Test adding negative numbers"""
        result = add_numbers(-2, -3)
        self.assertEqual(result, -5)
    
    def test_add_zero(self):
        """Test adding zero"""
        result = add_numbers(5, 0)
        self.assertEqual(result, 5)
    
    def test_add_mixed_signs(self):
        """Test adding positive and negative"""
        result = add_numbers(10, -3)
        self.assertEqual(result, 7)

# Run tests
if __name__ == '__main__':
    unittest.main()
```

### Common Test Assertions

```python
import unittest

class TestAssertions(unittest.TestCase):
    
    def test_equality(self):
        self.assertEqual(2 + 2, 4)
        self.assertNotEqual(2 + 2, 5)
    
    def test_boolean(self):
        self.assertTrue(5 > 3)
        self.assertFalse(5 < 3)
    
    def test_membership(self):
        self.assertIn('apple', ['apple', 'banana'])
        self.assertNotIn('grape', ['apple', 'banana'])
    
    def test_exceptions(self):
        with self.assertRaises(ZeroDivisionError):
            result = 10 / 0
    
    def test_approximate_equality(self):
        self.assertAlmostEqual(0.1 + 0.2, 0.3, places=7)
```

## Testing Data Analysis Functions

### Example 1: Testing Statistical Functions

```python
import unittest
import math

def calculate_statistics(data):
    """Calculate basic statistics for a dataset"""
    if not data:
        return None
    
    n = len(data)
    mean = sum(data) / n
    
    # Calculate standard deviation
    variance = sum((x - mean) ** 2 for x in data) / n
    std_dev = math.sqrt(variance)
    
    return {
        'count': n,
        'mean': mean,
        'std_dev': std_dev,
        'min': min(data),
        'max': max(data)
    }

class TestStatistics(unittest.TestCase):
    
    def test_empty_data(self):
        """Test with empty dataset"""
        result = calculate_statistics([])
        self.assertIsNone(result)
    
    def test_single_value(self):
        """Test with single value"""
        result = calculate_statistics([5])
        expected = {
            'count': 1,
            'mean': 5,
            'std_dev': 0,
            'min': 5,
            'max': 5
        }
        self.assertEqual(result, expected)
    
    def test_known_values(self):
        """Test with known statistical values"""
        data = [1, 2, 3, 4, 5]
        result = calculate_statistics(data)
        
        self.assertEqual(result['count'], 5)
        self.assertEqual(result['mean'], 3.0)
        self.assertEqual(result['min'], 1)
        self.assertEqual(result['max'], 5)
        self.assertAlmostEqual(result['std_dev'], 1.4142135623730951)
    
    def test_negative_numbers(self):
        """Test with negative numbers"""
        data = [-2, -1, 0, 1, 2]
        result = calculate_statistics(data)
        
        self.assertEqual(result['mean'], 0.0)
        self.assertEqual(result['min'], -2)
        self.assertEqual(result['max'], 2)

if __name__ == '__main__':
    unittest.main()
```

### Example 2: Testing Data Cleaning Functions

```python
import unittest

def clean_data(data):
    """Clean a list of data by removing invalid entries"""
    if not isinstance(data, list):
        raise TypeError("Data must be a list")
    
    cleaned = []
    for item in data:
        # Remove None values and empty strings
        if item is not None and item != "":
            # Convert strings to numbers if possible
            if isinstance(item, str):
                try:
                    item = float(item)
                except ValueError:
                    continue  # Skip non-numeric strings
            
            # Only keep numeric values
            if isinstance(item, (int, float)):
                cleaned.append(item)
    
    return cleaned

class TestDataCleaning(unittest.TestCase):
    
    def test_clean_mixed_data(self):
        """Test cleaning mixed data types"""
        dirty_data = [1, "2", 3.5, None, "", "invalid", "4.5", 0]
        result = clean_data(dirty_data)
        expected = [1, 2.0, 3.5, 4.5, 0]
        self.assertEqual(result, expected)
    
    def test_empty_list(self):
        """Test with empty list"""
        result = clean_data([])
        self.assertEqual(result, [])
    
    def test_all_invalid_data(self):
        """Test with all invalid data"""
        dirty_data = [None, "", "invalid", "not_a_number"]
        result = clean_data(dirty_data)
        self.assertEqual(result, [])
    
    def test_invalid_input_type(self):
        """Test with invalid input type"""
        with self.assertRaises(TypeError):
            clean_data("not a list")
    
    def test_all_valid_numbers(self):
        """Test with all valid numbers"""
        data = [1, 2, 3, 4, 5]
        result = clean_data(data)
        self.assertEqual(result, data)

if __name__ == '__main__':
    unittest.main()
```

## Test-Driven Development (TDD)

### The TDD Process
1. **Red**: Write a failing test
2. **Green**: Write minimal code to make it pass
3. **Refactor**: Improve the code while keeping tests passing

### TDD Example: Building a Grade Calculator

```python
import unittest

# Step 1: Write the test first (it will fail)
class TestGradeCalculator(unittest.TestCase):
    
    def test_calculate_letter_grade_A(self):
        """Test A grade calculation"""
        result = calculate_letter_grade(95)
        self.assertEqual(result, 'A')
    
    def test_calculate_letter_grade_B(self):
        """Test B grade calculation"""
        result = calculate_letter_grade(85)
        self.assertEqual(result, 'B')
    
    def test_calculate_letter_grade_F(self):
        """Test F grade calculation"""
        result = calculate_letter_grade(55)
        self.assertEqual(result, 'F')

# Step 2: Write minimal code to make tests pass
def calculate_letter_grade(percentage):
    """Convert percentage to letter grade"""
    if percentage >= 90:
        return 'A'
    elif percentage >= 80:
        return 'B'
    elif percentage >= 70:
        return 'C'
    elif percentage >= 60:
        return 'D'
    else:
        return 'F'

# Step 3: Add more tests and refactor as needed
class TestGradeCalculatorExtended(unittest.TestCase):
    
    def test_boundary_values(self):
        """Test boundary values"""
        self.assertEqual(calculate_letter_grade(90), 'A')
        self.assertEqual(calculate_letter_grade(89), 'B')
        self.assertEqual(calculate_letter_grade(80), 'B')
        self.assertEqual(calculate_letter_grade(79), 'C')
    
    def test_invalid_input(self):
        """Test invalid input handling"""
        with self.assertRaises(ValueError):
            calculate_letter_grade(-5)
        
        with self.assertRaises(ValueError):
            calculate_letter_grade(105)

# Update function to handle edge cases
def calculate_letter_grade(percentage):
    """Convert percentage to letter grade"""
    if not isinstance(percentage, (int, float)):
        raise TypeError("Percentage must be a number")
    
    if percentage < 0 or percentage > 100:
        raise ValueError("Percentage must be between 0 and 100")
    
    if percentage >= 90:
        return 'A'
    elif percentage >= 80:
        return 'B'
    elif percentage >= 70:
        return 'C'
    elif percentage >= 60:
        return 'D'
    else:
        return 'F'
```

## Testing File Operations

### Testing Functions that Read Files

```python
import unittest
import tempfile
import os

def count_lines_in_file(filename):
    """Count number of lines in a file"""
    try:
        with open(filename, 'r') as file:
            return len(file.readlines())
    except FileNotFoundError:
        return -1

class TestFileOperations(unittest.TestCase):
    
    def setUp(self):
        """Create temporary test files"""
        # Create temporary file with known content
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
        self.temp_file.write("Line 1\nLine 2\nLine 3\n")
        self.temp_file.close()
    
    def tearDown(self):
        """Clean up temporary files"""
        os.unlink(self.temp_file.name)
    
    def test_count_lines_existing_file(self):
        """Test counting lines in existing file"""
        result = count_lines_in_file(self.temp_file.name)
        self.assertEqual(result, 3)
    
    def test_count_lines_nonexistent_file(self):
        """Test counting lines in non-existent file"""
        result = count_lines_in_file("nonexistent_file.txt")
        self.assertEqual(result, -1)

if __name__ == '__main__':
    unittest.main()
```

## Mocking and Test Doubles

### Using unittest.mock for External Dependencies

```python
import unittest
from unittest.mock import patch, mock_open
import requests

def get_weather_data(city):
    """Get weather data from API"""
    url = f"http://api.weather.com/v1/current?city={city}"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        return {
            'temperature': data['temp'],
            'humidity': data['humidity'],
            'city': city
        }
    else:
        return None

class TestWeatherAPI(unittest.TestCase):
    
    @patch('requests.get')
    def test_successful_weather_request(self, mock_get):
        """Test successful API request"""
        # Mock the API response
        mock_response = unittest.mock.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'temp': 72,
            'humidity': 65
        }
        mock_get.return_value = mock_response
        
        # Test the function
        result = get_weather_data("Seattle")
        
        expected = {
            'temperature': 72,
            'humidity': 65,
            'city': 'Seattle'
        }
        self.assertEqual(result, expected)
    
    @patch('requests.get')
    def test_failed_weather_request(self, mock_get):
        """Test failed API request"""
        # Mock failed response
        mock_response = unittest.mock.Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        
        result = get_weather_data("InvalidCity")
        self.assertIsNone(result)

if __name__ == '__main__':
    unittest.main()
```

## Debugging Common Data Analysis Errors

### 1. Index Errors
```python
def safe_get_item(data, index):
    """Safely get item from list with debugging"""
    print(f"Accessing index {index} from list of length {len(data)}")
    
    if index < 0:
        print(f"Negative index {index} converted to {len(data) + index}")
        index = len(data) + index
    
    if 0 <= index < len(data):
        return data[index]
    else:
        print(f"Index {index} is out of range for list of length {len(data)}")
        return None

# Test with debugging
data = [10, 20, 30]
result = safe_get_item(data, 5)  # Will show debug info
```

### 2. Type Errors in Calculations
```python
def debug_calculation(values):
    """Debug calculation with type checking"""
    print(f"Input values: {values}")
    print(f"Input types: {[type(v) for v in values]}")
    
    numeric_values = []
    for i, value in enumerate(values):
        try:
            numeric_value = float(value)
            numeric_values.append(numeric_value)
            print(f"Converted {value} to {numeric_value}")
        except (ValueError, TypeError) as e:
            print(f"Could not convert {value} at index {i}: {e}")
    
    if numeric_values:
        result = sum(numeric_values) / len(numeric_values)
        print(f"Average of {numeric_values} = {result}")
        return result
    else:
        print("No valid numeric values found")
        return None

# Test with mixed data
mixed_data = [1, "2", 3.5, "invalid", None, "4"]
average = debug_calculation(mixed_data)
```

### 3. Logic Errors in Conditions
```python
def debug_grade_classification(scores):
    """Debug grade classification logic"""
    results = {}
    
    for student, score in scores.items():
        print(f"Processing {student} with score {score}")
        
        # Debug each condition
        if score >= 90:
            grade = 'A'
            print(f"  {score} >= 90: True -> Grade A")
        elif score >= 80:
            grade = 'B'
            print(f"  {score} >= 80: True -> Grade B")
        elif score >= 70:
            grade = 'C'
            print(f"  {score} >= 70: True -> Grade C")
        elif score >= 60:
            grade = 'D'
            print(f"  {score} >= 60: True -> Grade D")
        else:
            grade = 'F'
            print(f"  {score} < 60: True -> Grade F")
        
        results[student] = grade
    
    return results

# Test with debug output
student_scores = {"Alice": 85, "Bob": 92, "Charlie": 58}
grades = debug_grade_classification(student_scores)
```

## Best Practices for Testing

### 1. Test Structure (AAA Pattern)
```python
def test_calculate_discount(self):
    # Arrange: Set up test data
    original_price = 100
    discount_percent = 20
    
    # Act: Execute the function
    result = calculate_discount(original_price, discount_percent)
    
    # Assert: Check the result
    self.assertEqual(result, 80)
```

### 2. Descriptive Test Names
```python
class TestPriceCalculator(unittest.TestCase):
    
    # Good: Descriptive names
    def test_calculate_discount_with_valid_percentage(self):
        pass
    
    def test_calculate_discount_with_zero_percentage(self):
        pass
    
    def test_calculate_discount_raises_error_for_negative_price(self):
        pass
    
    # Avoid: Unclear names
    def test1(self):
        pass
    
    def test_discount(self):
        pass
```

### 3. Test Edge Cases
```python
class TestDataProcessor(unittest.TestCase):
    
    def test_empty_input(self):
        """Test with empty data"""
        pass
    
    def test_single_item_input(self):
        """Test with single item"""
        pass
    
    def test_large_input(self):
        """Test with large dataset"""
        pass
    
    def test_invalid_input_types(self):
        """Test with wrong data types"""
        pass
    
    def test_boundary_values(self):
        """Test with boundary values"""
        pass
```

## Debugging Tools and Techniques

### 1. Using IDE Debuggers
Most IDEs (VS Code, PyCharm) have built-in debuggers that let you:
- Set breakpoints
- Step through code line by line
- Inspect variable values
- Evaluate expressions

### 2. Python Debugger (pdb) Commands
```python
import pdb

def complex_function(data):
    pdb.set_trace()  # Debugger will stop here
    
    # Common pdb commands:
    # n (next line)
    # s (step into function)
    # c (continue execution)
    # l (list current code)
    # p variable_name (print variable)
    # pp variable_name (pretty print)
    # h (help)
    # q (quit debugger)
    
    result = process_data(data)
    return result
```

### 3. Assertion-Based Debugging
```python
def calculate_percentage(part, whole):
    # Use assertions to catch logic errors early
    assert isinstance(part, (int, float)), f"Part must be numeric, got {type(part)}"
    assert isinstance(whole, (int, float)), f"Whole must be numeric, got {type(whole)}"
    assert whole != 0, "Cannot divide by zero"
    assert part >= 0, f"Part cannot be negative, got {part}"
    assert whole > 0, f"Whole must be positive, got {whole}"
    
    percentage = (part / whole) * 100
    
    # Post-condition assertion
    assert 0 <= percentage <= 100, f"Percentage should be 0-100, got {percentage}"
    
    return percentage
```

## Key Terminology

- **Unit Test**: Test that verifies a single function or method
- **Test Case**: A single test scenario
- **Test Suite**: Collection of test cases
- **Assertion**: Statement that checks if a condition is true
- **Mock**: Fake object that simulates real object behavior
- **Test-Driven Development (TDD)**: Writing tests before writing code
- **Debugging**: Process of finding and fixing errors
- **Breakpoint**: Point where debugger pauses execution
- **Stack Trace**: List of function calls leading to an error

## Looking Ahead

In Lesson 08, we'll learn about:
- **Virtual Environments**: Isolating project dependencies
- **Package Management**: Installing and managing Python packages
- **pip**: Python's package installer
- **Requirements files**: Specifying project dependencies
- **Best practices**: Managing Python projects professionally

This completes the foundational phase of our Python course. You now have solid programming fundamentals, and we'll move into more specialized topics for data analysis!
