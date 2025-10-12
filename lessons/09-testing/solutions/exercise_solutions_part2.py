# Lesson 07 Solutions: Testing and Debugging - Part 2

import unittest
from unittest.mock import Mock, patch, mock_open
import tempfile
import os
import json

print("=== ADVANCED EXERCISE SOLUTIONS ===\n")

# Exercise 5: Data Validation Test Suite
print("Exercise 5: Data Validation Test Suite")
print("-" * 40)

import re
from datetime import datetime

def validate_email(email):
    """Validate email format"""
    if not email:
        return False
    
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def validate_phone(phone):
    """Validate phone number format (US format)"""
    if not phone:
        return False
    
    # Remove all non-digit characters
    digits = re.sub(r'\D', '', phone)
    
    # Check if it's 10 digits (US format)
    return len(digits) == 10

def validate_credit_card(card_number):
    """Validate credit card number using Luhn algorithm"""
    if not card_number:
        return False
    
    # Remove spaces and hyphens
    card_number = re.sub(r'[\s-]', '', card_number)
    
    # Check if all characters are digits
    if not card_number.isdigit():
        return False
    
    # Check length (13-19 digits for most cards)
    if len(card_number) < 13 or len(card_number) > 19:
        return False
    
    # Luhn algorithm
    def luhn_check(card_num):
        digits = [int(d) for d in card_num]
        for i in range(len(digits) - 2, -1, -2):
            digits[i] *= 2
            if digits[i] > 9:
                digits[i] -= 9
        return sum(digits) % 10 == 0
    
    return luhn_check(card_number)

def validate_date_format(date_string, format_string="%Y-%m-%d"):
    """Validate date format"""
    if not date_string:
        return False
    
    try:
        datetime.strptime(date_string, format_string)
        return True
    except ValueError:
        return False

class TestDataValidation(unittest.TestCase):
    def test_validate_email_valid(self):
        """Test valid email addresses"""
        valid_emails = [
            "user@example.com",
            "test.email@domain.org",
            "user+tag@example.co.uk",
            "123@example.com"
        ]
        for email in valid_emails:
            with self.subTest(email=email):
                self.assertTrue(validate_email(email))
    
    def test_validate_email_invalid(self):
        """Test invalid email addresses"""
        invalid_emails = [
            "",
            "invalid",
            "@example.com",
            "user@",
            "user@.com",
            "user..name@example.com",
            "user@example",
            "user name@example.com"
        ]
        for email in invalid_emails:
            with self.subTest(email=email):
                self.assertFalse(validate_email(email))
    
    def test_validate_phone_valid(self):
        """Test valid phone numbers"""
        valid_phones = [
            "1234567890",
            "(123) 456-7890",
            "123-456-7890",
            "123.456.7890",
            "+1 123 456 7890"
        ]
        for phone in valid_phones:
            with self.subTest(phone=phone):
                self.assertTrue(validate_phone(phone))
    
    def test_validate_phone_invalid(self):
        """Test invalid phone numbers"""
        invalid_phones = [
            "",
            "123456789",  # Too short
            "12345678901",  # Too long
            "abcdefghij",  # Letters
            "123-456-789a"  # Mixed
        ]
        for phone in invalid_phones:
            with self.subTest(phone=phone):
                self.assertFalse(validate_phone(phone))
    
    def test_validate_credit_card_valid(self):
        """Test valid credit card numbers"""
        valid_cards = [
            "4532015112830366",  # Visa
            "5555555555554444",  # MasterCard
            "378282246310005",   # American Express
            "4532-0151-1283-0366"  # With hyphens
        ]
        for card in valid_cards:
            with self.subTest(card=card):
                self.assertTrue(validate_credit_card(card))
    
    def test_validate_credit_card_invalid(self):
        """Test invalid credit card numbers"""
        invalid_cards = [
            "",
            "1234567890123456",  # Fails Luhn check
            "123456789012345a",  # Contains letter
            "123456789012",      # Too short
            "12345678901234567890"  # Too long
        ]
        for card in invalid_cards:
            with self.subTest(card=card):
                self.assertFalse(validate_credit_card(card))
    
    def test_validate_date_format_valid(self):
        """Test valid date formats"""
        valid_dates = [
            "2024-01-15",
            "2023-12-31",
            "2024-02-29"  # Leap year
        ]
        for date in valid_dates:
            with self.subTest(date=date):
                self.assertTrue(validate_date_format(date))
    
    def test_validate_date_format_invalid(self):
        """Test invalid date formats"""
        invalid_dates = [
            "",
            "2024-13-01",  # Invalid month
            "2024-02-30",  # Invalid day
            "24-01-15",    # Wrong format
            "2024/01/15",  # Wrong separator
            "not-a-date"
        ]
        for date in invalid_dates:
            with self.subTest(date=date):
                self.assertFalse(validate_date_format(date))

print("Data validation test suite completed")
print()

# Exercise 6: File Operations Testing
print("Exercise 6: File Operations Testing")
print("-" * 40)

def read_config_file(filename):
    """Read JSON config file"""
    try:
        with open(filename, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file {filename} not found")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in config file {filename}")

def save_user_data(user_data, filename):
    """Save user data to CSV"""
    import csv
    
    if not user_data:
        raise ValueError("No user data provided")
    
    with open(filename, 'w', newline='') as file:
        if isinstance(user_data, list) and len(user_data) > 0:
            writer = csv.DictWriter(file, fieldnames=user_data[0].keys())
            writer.writeheader()
            writer.writerows(user_data)
        else:
            raise ValueError("User data must be a non-empty list of dictionaries")

def backup_file(source, destination):
    """Create file backup"""
    import shutil
    
    if not os.path.exists(source):
        raise FileNotFoundError(f"Source file {source} not found")
    
    try:
        shutil.copy2(source, destination)
        return True
    except PermissionError:
        raise PermissionError(f"Permission denied copying to {destination}")

def count_lines_in_file(filename):
    """Count lines in text file"""
    try:
        with open(filename, 'r') as file:
            return sum(1 for line in file)
    except FileNotFoundError:
        raise FileNotFoundError(f"File {filename} not found")

class TestFileOperations(unittest.TestCase):
    def setUp(self):
        """Set up temporary files for testing"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test config file
        self.config_file = os.path.join(self.temp_dir, "config.json")
        config_data = {"database": {"host": "localhost", "port": 5432}}
        with open(self.config_file, 'w') as f:
            json.dump(config_data, f)
        
        # Create test text file
        self.text_file = os.path.join(self.temp_dir, "test.txt")
        with open(self.text_file, 'w') as f:
            f.write("Line 1\nLine 2\nLine 3\n")
    
    def tearDown(self):
        """Clean up temporary files"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_read_config_file_success(self):
        """Test successful config file reading"""
        config = read_config_file(self.config_file)
        self.assertIn("database", config)
        self.assertEqual(config["database"]["host"], "localhost")
    
    def test_read_config_file_not_found(self):
        """Test config file not found"""
        with self.assertRaises(FileNotFoundError):
            read_config_file("nonexistent.json")
    
    def test_read_config_file_invalid_json(self):
        """Test invalid JSON in config file"""
        invalid_json_file = os.path.join(self.temp_dir, "invalid.json")
        with open(invalid_json_file, 'w') as f:
            f.write("{ invalid json }")
        
        with self.assertRaises(ValueError):
            read_config_file(invalid_json_file)
    
    def test_save_user_data_success(self):
        """Test successful user data saving"""
        user_data = [
            {"name": "Alice", "age": 25},
            {"name": "Bob", "age": 30}
        ]
        csv_file = os.path.join(self.temp_dir, "users.csv")
        
        save_user_data(user_data, csv_file)
        
        # Verify file was created and has correct content
        self.assertTrue(os.path.exists(csv_file))
        with open(csv_file, 'r') as f:
            content = f.read()
            self.assertIn("name,age", content)
            self.assertIn("Alice,25", content)
    
    def test_save_user_data_empty_data(self):
        """Test saving empty user data"""
        csv_file = os.path.join(self.temp_dir, "empty.csv")
        
        with self.assertRaises(ValueError):
            save_user_data([], csv_file)
    
    def test_backup_file_success(self):
        """Test successful file backup"""
        backup_file_path = os.path.join(self.temp_dir, "backup.txt")
        
        result = backup_file(self.text_file, backup_file_path)
        
        self.assertTrue(result)
        self.assertTrue(os.path.exists(backup_file_path))
        
        # Verify content is the same
        with open(self.text_file, 'r') as f1, open(backup_file_path, 'r') as f2:
            self.assertEqual(f1.read(), f2.read())
    
    def test_backup_file_source_not_found(self):
        """Test backup when source file doesn't exist"""
        backup_path = os.path.join(self.temp_dir, "backup.txt")
        
        with self.assertRaises(FileNotFoundError):
            backup_file("nonexistent.txt", backup_path)
    
    def test_count_lines_in_file_success(self):
        """Test successful line counting"""
        count = count_lines_in_file(self.text_file)
        self.assertEqual(count, 3)
    
    def test_count_lines_in_file_not_found(self):
        """Test line counting when file doesn't exist"""
        with self.assertRaises(FileNotFoundError):
            count_lines_in_file("nonexistent.txt")
    
    def test_count_lines_empty_file(self):
        """Test line counting for empty file"""
        empty_file = os.path.join(self.temp_dir, "empty.txt")
        with open(empty_file, 'w') as f:
            pass  # Create empty file
        
        count = count_lines_in_file(empty_file)
        self.assertEqual(count, 0)

print("File operations test suite completed")
print()

# Exercise 7: Shopping Cart Test Suite
print("Exercise 7: Shopping Cart Test Suite")
print("-" * 40)

class ShoppingCart:
    """Shopping cart implementation for testing"""
    
    def __init__(self):
        self.items = {}  # {item_name: {'price': float, 'quantity': int}}
        self.discount_percentage = 0
    
    def add_item(self, item, price, quantity=1):
        """Add item to cart"""
        if not item or price < 0 or quantity <= 0:
            raise ValueError("Invalid item, price, or quantity")
        
        if item in self.items:
            self.items[item]['quantity'] += quantity
        else:
            self.items[item] = {'price': price, 'quantity': quantity}
    
    def remove_item(self, item):
        """Remove item from cart"""
        if item not in self.items:
            raise KeyError(f"Item '{item}' not in cart")
        
        del self.items[item]
    
    def update_quantity(self, item, new_quantity):
        """Update item quantity"""
        if item not in self.items:
            raise KeyError(f"Item '{item}' not in cart")
        
        if new_quantity <= 0:
            raise ValueError("Quantity must be positive")
        
        self.items[item]['quantity'] = new_quantity
    
    def calculate_total(self):
        """Calculate total cost"""
        subtotal = sum(item['price'] * item['quantity'] for item in self.items.values())
        discount_amount = subtotal * (self.discount_percentage / 100)
        return subtotal - discount_amount
    
    def apply_discount(self, percentage):
        """Apply discount percentage"""
        if percentage < 0 or percentage > 100:
            raise ValueError("Discount percentage must be between 0 and 100")
        
        self.discount_percentage = percentage
    
    def get_item_count(self):
        """Get total number of items"""
        return sum(item['quantity'] for item in self.items.values())
    
    def clear_cart(self):
        """Clear all items from cart"""
        self.items.clear()
        self.discount_percentage = 0

class TestShoppingCart(unittest.TestCase):
    def setUp(self):
        """Set up fresh cart for each test"""
        self.cart = ShoppingCart()
    
    def test_add_item_new(self):
        """Test adding new item to cart"""
        self.cart.add_item("Apple", 1.50, 3)
        
        self.assertIn("Apple", self.cart.items)
        self.assertEqual(self.cart.items["Apple"]["price"], 1.50)
        self.assertEqual(self.cart.items["Apple"]["quantity"], 3)
    
    def test_add_item_existing(self):
        """Test adding to existing item"""
        self.cart.add_item("Apple", 1.50, 2)
        self.cart.add_item("Apple", 1.50, 3)
        
        self.assertEqual(self.cart.items["Apple"]["quantity"], 5)
    
    def test_add_item_invalid_inputs(self):
        """Test adding item with invalid inputs"""
        with self.assertRaises(ValueError):
            self.cart.add_item("", 1.50, 1)  # Empty name
        
        with self.assertRaises(ValueError):
            self.cart.add_item("Apple", -1.50, 1)  # Negative price
        
        with self.assertRaises(ValueError):
            self.cart.add_item("Apple", 1.50, 0)  # Zero quantity
    
    def test_remove_item_success(self):
        """Test successful item removal"""
        self.cart.add_item("Apple", 1.50, 2)
        self.cart.remove_item("Apple")
        
        self.assertNotIn("Apple", self.cart.items)
    
    def test_remove_item_not_found(self):
        """Test removing non-existent item"""
        with self.assertRaises(KeyError):
            self.cart.remove_item("NonExistent")
    
    def test_update_quantity_success(self):
        """Test successful quantity update"""
        self.cart.add_item("Apple", 1.50, 2)
        self.cart.update_quantity("Apple", 5)
        
        self.assertEqual(self.cart.items["Apple"]["quantity"], 5)
    
    def test_update_quantity_item_not_found(self):
        """Test updating quantity for non-existent item"""
        with self.assertRaises(KeyError):
            self.cart.update_quantity("NonExistent", 5)
    
    def test_update_quantity_invalid(self):
        """Test updating with invalid quantity"""
        self.cart.add_item("Apple", 1.50, 2)
        
        with self.assertRaises(ValueError):
            self.cart.update_quantity("Apple", 0)
        
        with self.assertRaises(ValueError):
            self.cart.update_quantity("Apple", -1)
    
    def test_calculate_total_no_discount(self):
        """Test total calculation without discount"""
        self.cart.add_item("Apple", 1.50, 2)  # 3.00
        self.cart.add_item("Banana", 0.75, 4)  # 3.00
        
        total = self.cart.calculate_total()
        self.assertEqual(total, 6.00)
    
    def test_calculate_total_with_discount(self):
        """Test total calculation with discount"""
        self.cart.add_item("Apple", 10.00, 1)
        self.cart.apply_discount(20)  # 20% discount
        
        total = self.cart.calculate_total()
        self.assertEqual(total, 8.00)  # 10.00 - 2.00
    
    def test_apply_discount_valid(self):
        """Test applying valid discount"""
        self.cart.apply_discount(15)
        self.assertEqual(self.cart.discount_percentage, 15)
    
    def test_apply_discount_invalid(self):
        """Test applying invalid discount"""
        with self.assertRaises(ValueError):
            self.cart.apply_discount(-5)
        
        with self.assertRaises(ValueError):
            self.cart.apply_discount(105)
    
    def test_get_item_count(self):
        """Test getting total item count"""
        self.cart.add_item("Apple", 1.50, 3)
        self.cart.add_item("Banana", 0.75, 2)
        
        count = self.cart.get_item_count()
        self.assertEqual(count, 5)
    
    def test_get_item_count_empty_cart(self):
        """Test item count for empty cart"""
        count = self.cart.get_item_count()
        self.assertEqual(count, 0)
    
    def test_clear_cart(self):
        """Test clearing cart"""
        self.cart.add_item("Apple", 1.50, 3)
        self.cart.apply_discount(10)
        
        self.cart.clear_cart()
        
        self.assertEqual(len(self.cart.items), 0)
        self.assertEqual(self.cart.discount_percentage, 0)

print("Shopping cart test suite completed")
print()

if __name__ == "__main__":
    # Run the tests
    unittest.main(verbosity=2, exit=False)
