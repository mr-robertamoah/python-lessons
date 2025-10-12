# Lesson 07 Exercises: Testing and Debugging

## Guided Exercises (Do with Instructor)

### Exercise 1: Writing Your First Unit Tests
**Goal**: Create basic unit tests using unittest

**Tasks**:
1. Create a simple calculator function
2. Write tests for basic operations
3. Run tests and interpret results
4. Fix any failing tests

```python
import unittest

def calculator(a, b, operation):
    """Simple calculator function"""
    # Implement basic operations
    pass

class TestCalculator(unittest.TestCase):
    def test_addition(self):
        # Write test for addition
        pass
    
    def test_division_by_zero(self):
        # Write test for division by zero
        pass
```

---

### Exercise 2: Test-Driven Development (TDD)
**Goal**: Practice writing tests before code

**Tasks**:
1. Write tests for a password validator function (before implementing)
2. Run tests (they should fail)
3. Implement the function to make tests pass
4. Refactor and ensure tests still pass

**Password Requirements**:
- At least 8 characters long
- Contains uppercase and lowercase letters
- Contains at least one digit
- Contains at least one special character

---

### Exercise 3: Debugging with Print Statements
**Goal**: Learn systematic debugging techniques

**Tasks**:
1. Debug a buggy function using print statements
2. Identify the root cause of the bug
3. Fix the bug and verify the solution
4. Remove debug print statements

```python
def find_average_grade(students):
    """Find average grade - contains bugs!"""
    total = 0
    for student in students:
        total += student['grade']
    return total / len(students)

# Test data that reveals the bug
test_students = [
    {'name': 'Alice', 'grade': 85},
    {'name': 'Bob', 'grade': 92},
    {'name': 'Carol'},  # Missing grade!
    {'name': 'David', 'grade': 78}
]
```

---

## Independent Exercises (Try on Your Own)

### Exercise 4: String Utilities Test Suite
**Goal**: Create comprehensive tests for string utility functions

**Functions to Test**:
1. `is_palindrome(text)` - Check if text reads same forwards/backwards
2. `word_count(text)` - Count words in text
3. `capitalize_words(text)` - Capitalize first letter of each word
4. `remove_duplicates(text)` - Remove duplicate words

**Requirements**:
- Test normal cases
- Test edge cases (empty strings, single characters)
- Test error conditions
- Use descriptive test method names

---

### Exercise 5: Data Validation Test Suite
**Goal**: Test data validation functions thoroughly

**Create tests for**:
1. Email validation function
2. Phone number validation function
3. Credit card number validation function
4. Date format validation function

**Test Cases to Include**:
- Valid inputs
- Invalid formats
- Edge cases (empty strings, None values)
- Boundary conditions

---

### Exercise 6: File Operations Testing
**Goal**: Test file handling functions with mocking

**Functions to Test**:
1. `read_config_file(filename)` - Read JSON config
2. `save_user_data(user_data, filename)` - Save user data to CSV
3. `backup_file(source, destination)` - Create file backup
4. `count_lines_in_file(filename)` - Count lines in text file

**Testing Challenges**:
- Test without creating actual files
- Test error conditions (file not found, permission denied)
- Test with different file contents

---

### Exercise 7: Shopping Cart Test Suite
**Goal**: Test a more complex class with multiple methods

**ShoppingCart Class Methods**:
- `add_item(item, price, quantity)`
- `remove_item(item)`
- `update_quantity(item, new_quantity)`
- `calculate_total()`
- `apply_discount(percentage)`
- `get_item_count()`

**Test Requirements**:
- Test each method individually
- Test method interactions
- Test edge cases and error conditions
- Use setUp and tearDown methods

---

## Challenge Exercises (If You Finish Early)

### Challenge 1: Bank Account System Testing
**Goal**: Test a complex financial system

**BankAccount Class Features**:
- Deposit and withdrawal methods
- Balance tracking
- Transaction history
- Overdraft protection
- Interest calculation
- Account freezing/unfreezing

**Advanced Testing Requirements**:
- Test transaction sequences
- Test concurrent operations (if applicable)
- Test business rules and constraints
- Mock external dependencies (like date/time)

---

### Challenge 2: Web API Response Testing
**Goal**: Test functions that process API responses

**Functions to Test**:
1. `parse_weather_data(api_response)` - Parse weather API JSON
2. `validate_api_response(response)` - Validate response structure
3. `extract_user_info(api_response)` - Extract user data
4. `handle_api_errors(response)` - Handle different error codes

**Testing Challenges**:
- Mock API responses
- Test different response formats
- Test error handling
- Test data transformation

---

### Challenge 3: Game Logic Testing
**Goal**: Test interactive game logic

**Tic-Tac-Toe Game Functions**:
- `make_move(board, player, position)`
- `check_winner(board)`
- `is_board_full(board)`
- `get_valid_moves(board)`
- `evaluate_position(board, player)`

**Testing Requirements**:
- Test all winning conditions
- Test draw conditions
- Test invalid moves
- Test game state transitions

---

## Debugging Exercises

### Exercise 8: Debug the Sorting Algorithm
**Goal**: Find and fix bugs in sorting implementations

**Buggy Code to Debug**:
```python
def bubble_sort(arr):
    """Bubble sort with bugs"""
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i):  # Bug here!
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

def quick_sort(arr):
    """Quick sort with bugs"""
    if len(arr) <= 1:
        return arr
    
    pivot = arr[0]
    less = [x for x in arr if x < pivot]
    equal = [x for x in arr if x == pivot]
    greater = [x for x in arr if x > pivot]
    
    return quick_sort(less) + equal + quick_sort(greater)  # Missing something?
```

---

### Exercise 9: Debug the Data Processing Pipeline
**Goal**: Debug a multi-step data processing function

**Buggy Pipeline**:
```python
def process_sales_data(raw_data):
    """Process sales data - contains multiple bugs"""
    # Step 1: Clean data
    cleaned_data = []
    for record in raw_data:
        if record['amount'] > 0:  # Bug: what about zero amounts?
            cleaned_data.append(record)
    
    # Step 2: Calculate totals
    total_sales = 0
    for record in cleaned_data:
        total_sales += record['amount']
    
    # Step 3: Group by category
    categories = {}
    for record in cleaned_data:
        category = record['category']
        if category in categories:
            categories[category] += record['amount']
        else:
            categories[category] = record['amount']
    
    # Step 4: Calculate percentages
    percentages = {}
    for category, amount in categories.items():
        percentages[category] = (amount / total_sales) * 100
    
    return {
        'total_sales': total_sales,
        'categories': categories,
        'percentages': percentages
    }
```

---

### Exercise 10: Debug the Recursive Function
**Goal**: Debug recursive functions with edge cases

**Buggy Recursive Functions**:
```python
def factorial(n):
    """Calculate factorial - has bugs"""
    if n == 1:  # Bug: what about n=0?
        return 1
    return n * factorial(n - 1)

def fibonacci(n):
    """Calculate fibonacci number - has bugs"""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)  # Inefficient, but correct?

def binary_search(arr, target, left=0, right=None):
    """Binary search - has bugs"""
    if right is None:
        right = len(arr)
    
    if left >= right:
        return -1
    
    mid = (left + right) // 2
    
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search(arr, target, mid + 1, right)
    else:
        return binary_search(arr, target, left, mid - 1)  # Bug here?
```

---

## Testing Best Practices Exercises

### Exercise 11: Test Organization and Structure
**Goal**: Learn to organize tests effectively

**Tasks**:
1. Create a test module structure for a library management system
2. Organize tests by functionality
3. Use test fixtures and setup methods
4. Create helper functions for common test operations

**Library System Components**:
- Book management
- Member management
- Checkout/return operations
- Fine calculations
- Report generation

---

### Exercise 12: Mocking and Test Doubles
**Goal**: Practice using mocks and test doubles

**Scenarios to Mock**:
1. Database connections
2. File system operations
3. Network requests
4. Current date/time
5. Random number generation

```python
from unittest.mock import Mock, patch, MagicMock

class DatabaseService:
    def get_user(self, user_id):
        # This would normally connect to database
        pass
    
    def save_user(self, user_data):
        # This would normally save to database
        pass

# Write tests that mock the database operations
```

---

### Exercise 13: Performance Testing
**Goal**: Test function performance and efficiency

**Tasks**:
1. Create performance tests for sorting algorithms
2. Test memory usage of data structures
3. Test time complexity of search operations
4. Create benchmarks for comparison

```python
import time
import tracemalloc

def performance_test(func, *args, **kwargs):
    """Measure function performance"""
    # Measure time
    start_time = time.time()
    
    # Measure memory
    tracemalloc.start()
    
    result = func(*args, **kwargs)
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    end_time = time.time()
    
    return {
        'result': result,
        'time': end_time - start_time,
        'memory_current': current,
        'memory_peak': peak
    }
```

---

## Real-World Testing Scenarios

### Exercise 14: E-commerce System Testing
**Goal**: Test a realistic e-commerce system

**System Components**:
- Product catalog
- Shopping cart
- Order processing
- Payment handling
- Inventory management
- User authentication

**Testing Requirements**:
- Unit tests for individual components
- Integration tests for component interactions
- End-to-end workflow tests
- Error handling and edge cases

---

### Exercise 15: Data Analysis Pipeline Testing
**Goal**: Test data processing and analysis functions

**Pipeline Components**:
1. Data loading and validation
2. Data cleaning and transformation
3. Statistical calculations
4. Data visualization preparation
5. Report generation

**Testing Challenges**:
- Test with different data formats
- Test with missing or invalid data
- Test statistical accuracy
- Test performance with large datasets

---

## Debugging Tools and Techniques

### Exercise 16: Using Python Debugger (pdb)
**Goal**: Learn to use the Python debugger effectively

**Tasks**:
1. Set breakpoints in code
2. Step through code execution
3. Inspect variable values
4. Modify variables during debugging
5. Use debugger commands effectively

```python
import pdb

def complex_calculation(data):
    """Complex function to debug"""
    pdb.set_trace()  # Set breakpoint here
    
    result = 0
    for item in data:
        if item > 0:
            result += item * 2
        else:
            result -= abs(item)
    
    return result / len(data)
```

---

### Exercise 17: Logging for Debugging
**Goal**: Use logging instead of print statements

**Tasks**:
1. Set up logging configuration
2. Use different log levels appropriately
3. Log function entry/exit
4. Log variable values and state changes
5. Create log analysis tools

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def process_data(data):
    """Function with proper logging"""
    logger.info(f"Processing {len(data)} items")
    
    for i, item in enumerate(data):
        logger.debug(f"Processing item {i}: {item}")
        # Process item
    
    logger.info("Processing complete")
```

---

## Self-Assessment Checklist

After completing these exercises, you should be able to:

- [ ] Write unit tests using unittest framework
- [ ] Practice test-driven development (TDD)
- [ ] Debug code using print statements and debugger
- [ ] Test edge cases and error conditions
- [ ] Use mocks and test doubles effectively
- [ ] Organize tests in a logical structure
- [ ] Write descriptive test names and documentation
- [ ] Test file operations and external dependencies
- [ ] Debug recursive and complex algorithms
- [ ] Use logging for debugging purposes
- [ ] Measure and test performance
- [ ] Handle test setup and teardown properly

## Testing Best Practices Summary

### Writing Good Tests
1. **Test one thing at a time**: Each test should focus on a single behavior
2. **Use descriptive names**: Test names should explain what is being tested
3. **Follow AAA pattern**: Arrange, Act, Assert
4. **Test edge cases**: Empty inputs, boundary values, error conditions
5. **Keep tests independent**: Tests shouldn't depend on each other

### Test Organization
1. **Group related tests**: Use test classes to group related functionality
2. **Use setup/teardown**: Initialize test data in setUp methods
3. **Create test utilities**: Helper functions for common test operations
4. **Separate test types**: Unit tests, integration tests, end-to-end tests

### Debugging Strategies
1. **Reproduce the bug**: Create minimal test case that shows the problem
2. **Use systematic approach**: Binary search to isolate the problem
3. **Check assumptions**: Verify that your assumptions about the code are correct
4. **Use appropriate tools**: Debugger, logging, profiling tools
5. **Fix root cause**: Don't just fix symptoms

## Common Testing Patterns

### Test Structure
```python
class TestClassName(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        pass
    
    def tearDown(self):
        """Clean up after each test method."""
        pass
    
    def test_specific_behavior(self):
        """Test a specific behavior - use descriptive names."""
        # Arrange
        # Act
        # Assert
        pass
```

### Assertion Methods
```python
# Equality
self.assertEqual(actual, expected)
self.assertNotEqual(actual, expected)

# Truth values
self.assertTrue(condition)
self.assertFalse(condition)

# None values
self.assertIsNone(value)
self.assertIsNotNone(value)

# Exceptions
self.assertRaises(ExceptionType, function, args)
with self.assertRaises(ExceptionType):
    # code that should raise exception

# Collections
self.assertIn(item, collection)
self.assertNotIn(item, collection)
```

## Git Reminder

Save your work:

1. Create folder `lesson-07-testing` in your repository
2. Save test files with descriptive names
3. Include both working and buggy code examples
4. Commit and push:
   ```bash
   git add .
   git commit -m "Complete Lesson 07: Testing and Debugging"
   git push
   ```

## Next Lesson Preview

In Lesson 08, we'll learn about:
- **Virtual Environments**: Isolating project dependencies
- **Package Management**: Using pip and requirements.txt
- **Project Structure**: Organizing larger Python projects
- **Environment Variables**: Managing configuration and secrets
