# Lesson 06 Solutions: File Input/Output and CSV Processing

import csv
import json
import os
from pathlib import Path
from datetime import datetime
import shutil

print("=== GUIDED EXERCISE SOLUTIONS ===\n")

# Exercise 1: Basic File Operations
print("Exercise 1: Basic File Operations")
print("-" * 40)

# Create file with quotes
quotes_file = "quotes.txt"
quotes = [
    "The only way to do great work is to love what you do. - Steve Jobs",
    "Innovation distinguishes between a leader and a follower. - Steve Jobs",
    "Life is what happens to you while you're busy making other plans. - John Lennon"
]

with open(quotes_file, 'w') as file:
    for quote in quotes:
        file.write(quote + '\n')

print(f"Created {quotes_file} with {len(quotes)} quotes")

# Read and display contents
print("\nFile contents:")
with open(quotes_file, 'r') as file:
    content = file.read()
    print(content)

# Append new quote
new_quote = "The future belongs to those who believe in the beauty of their dreams. - Eleanor Roosevelt"
with open(quotes_file, 'a') as file:
    file.write(new_quote + '\n')

print("Appended new quote")

# Count lines and words
line_count = 0
word_count = 0
with open(quotes_file, 'r') as file:
    for line in file:
        line_count += 1
        word_count += len(line.split())

print(f"File statistics: {line_count} lines, {word_count} words")

# Clean up
os.remove(quotes_file)
print(f"Cleaned up {quotes_file}")
print()

# Exercise 2: Error Handling with Files
print("Exercise 2: Error Handling with Files")
print("-" * 40)

def safe_file_read(filename):
    try:
        with open(filename, 'r') as file:
            return file.read()
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
        return None
    except PermissionError:
        print(f"Error: Permission denied for '{filename}'")
        return None
    except OSError as e:
        print(f"Error: OS error occurred - {e}")
        return None

def safe_file_write(filename, content):
    try:
        with open(filename, 'w') as file:
            file.write(content)
        print(f"Successfully wrote to {filename}")
        return True
    except PermissionError:
        print(f"Error: Permission denied for '{filename}'")
        return False
    except OSError as e:
        print(f"Error: OS error occurred - {e}")
        return False

# Test error handling
result = safe_file_read("nonexistent.txt")
print(f"Read result: {result}")

success = safe_file_write("test.txt", "Hello World")
if success:
    content = safe_file_read("test.txt")
    print(f"File content: {content}")
    os.remove("test.txt")
print()

# Exercise 3: CSV File Basics
print("Exercise 3: CSV File Basics")
print("-" * 40)

# Create CSV with student data
students_data = [
    ["Name", "Age", "Grade", "Score"],
    ["Alice Johnson", 20, "A", 95],
    ["Bob Smith", 19, "B", 87],
    ["Carol Wilson", 21, "A", 92],
    ["David Brown", 20, "C", 78],
    ["Eva Davis", 19, "B", 89]
]

csv_file = "students.csv"
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(students_data)

print(f"Created {csv_file}")

# Read and display as table
print("\nStudent Data Table:")
with open(csv_file, 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        print(f"{row[0]:<15} {row[1]:<5} {row[2]:<7} {row[3]:<5}")

# Add new student
new_student = ["Frank Miller", 22, "B", 85]
with open(csv_file, 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(new_student)

print(f"\nAdded new student: {new_student[0]}")

# Calculate averages
total_score = 0
count = 0
with open(csv_file, 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        total_score += int(row['Score'])
        count += 1

average_score = total_score / count if count > 0 else 0
print(f"Class average score: {average_score:.1f}")

# Clean up
os.remove(csv_file)
print(f"Cleaned up {csv_file}")
print()

print("=== INDEPENDENT EXERCISE SOLUTIONS ===\n")

# Exercise 4: Personal Journal Application
print("Exercise 4: Personal Journal Application")
print("-" * 40)

class PersonalJournal:
    def __init__(self, journal_file="journal.txt"):
        self.journal_file = journal_file
    
    def add_entry(self, entry_text):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.journal_file, 'a') as file:
            file.write(f"[{timestamp}] {entry_text}\n")
        print(f"Added journal entry at {timestamp}")
    
    def read_entries(self):
        try:
            with open(self.journal_file, 'r') as file:
                entries = file.readlines()
            return entries
        except FileNotFoundError:
            return []
    
    def search_entries(self, keyword):
        entries = self.read_entries()
        matching_entries = []
        for entry in entries:
            if keyword.lower() in entry.lower():
                matching_entries.append(entry.strip())
        return matching_entries
    
    def display_entries(self):
        entries = self.read_entries()
        if not entries:
            print("No journal entries found.")
            return
        
        print("Journal Entries:")
        for entry in entries:
            print(f"  {entry.strip()}")

# Demo the journal
journal = PersonalJournal()
journal.add_entry("Started learning Python file I/O today. Very interesting!")
journal.add_entry("Practiced CSV operations. Getting more comfortable with data handling.")
journal.add_entry("Working on file error handling. Important for robust applications.")

journal.display_entries()

search_results = journal.search_entries("Python")
print(f"\nEntries containing 'Python': {len(search_results)}")
for result in search_results:
    print(f"  {result}")

# Clean up
if os.path.exists("journal.txt"):
    os.remove("journal.txt")
print("Cleaned up journal file")
print()

# Exercise 5: Grade Book File Manager
print("Exercise 5: Grade Book File Manager")
print("-" * 40)

class GradeBookManager:
    def __init__(self, csv_file="gradebook.csv"):
        self.csv_file = csv_file
        self.students = []
        self.load_data()
    
    def load_data(self):
        try:
            with open(self.csv_file, 'r') as file:
                reader = csv.DictReader(file)
                self.students = list(reader)
        except FileNotFoundError:
            self.students = []
            print(f"No existing gradebook found. Starting fresh.")
    
    def save_data(self):
        if not self.students:
            return
        
        fieldnames = self.students[0].keys()
        with open(self.csv_file, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.students)
        print(f"Saved gradebook to {self.csv_file}")
    
    def add_student(self, student_data):
        self.students.append(student_data)
        print(f"Added student: {student_data['Name']}")
    
    def calculate_statistics(self):
        if not self.students:
            return {}
        
        subjects = [key for key in self.students[0].keys() if key not in ['Student_ID', 'Name']]
        stats = {}
        
        for subject in subjects:
            scores = [int(student[subject]) for student in self.students]
            stats[subject] = {
                'average': sum(scores) / len(scores),
                'highest': max(scores),
                'lowest': min(scores)
            }
        
        return stats
    
    def export_report(self, report_file="grade_report.csv"):
        stats = self.calculate_statistics()
        
        with open(report_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Subject", "Average", "Highest", "Lowest"])
            
            for subject, data in stats.items():
                writer.writerow([
                    subject,
                    f"{data['average']:.1f}",
                    data['highest'],
                    data['lowest']
                ])
        
        print(f"Exported report to {report_file}")

# Demo the grade book manager
gradebook = GradeBookManager()

# Add sample students
sample_students = [
    {"Student_ID": "001", "Name": "Alice Johnson", "Math": "85", "Science": "92", "English": "78", "History": "88"},
    {"Student_ID": "002", "Name": "Bob Smith", "Math": "76", "Science": "84", "English": "82", "History": "79"},
    {"Student_ID": "003", "Name": "Carol Wilson", "Math": "94", "Science": "89", "English": "91", "History": "87"}
]

for student in sample_students:
    gradebook.add_student(student)

gradebook.save_data()

# Calculate and display statistics
stats = gradebook.calculate_statistics()
print("\nClass Statistics:")
for subject, data in stats.items():
    print(f"{subject}: Avg={data['average']:.1f}, High={data['highest']}, Low={data['lowest']}")

gradebook.export_report()

# Clean up
for file in ["gradebook.csv", "grade_report.csv"]:
    if os.path.exists(file):
        os.remove(file)
print("Cleaned up gradebook files")
print()

# Exercise 6: Inventory System with File Storage
print("Exercise 6: Inventory System with File Storage")
print("-" * 40)

class InventorySystem:
    def __init__(self, inventory_file="inventory.csv", log_file="transactions.log"):
        self.inventory_file = inventory_file
        self.log_file = log_file
        self.inventory = {}
        self.load_inventory()
    
    def load_inventory(self):
        try:
            with open(self.inventory_file, 'r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    self.inventory[row['Product_ID']] = {
                        'name': row['Name'],
                        'quantity': int(row['Quantity']),
                        'price': float(row['Price']),
                        'supplier': row['Supplier']
                    }
        except FileNotFoundError:
            print("No existing inventory found. Starting fresh.")
    
    def save_inventory(self):
        with open(self.inventory_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Product_ID', 'Name', 'Quantity', 'Price', 'Supplier'])
            
            for pid, product in self.inventory.items():
                writer.writerow([
                    pid,
                    product['name'],
                    product['quantity'],
                    product['price'],
                    product['supplier']
                ])
    
    def log_transaction(self, transaction_type, product_id, quantity, notes=""):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, 'a') as file:
            file.write(f"{timestamp},{transaction_type},{product_id},{quantity},{notes}\n")
    
    def add_stock(self, product_id, quantity):
        if product_id in self.inventory:
            self.inventory[product_id]['quantity'] += quantity
            self.log_transaction("ADD", product_id, quantity)
            print(f"Added {quantity} units of {self.inventory[product_id]['name']}")
        else:
            print("Product not found")
    
    def remove_stock(self, product_id, quantity):
        if product_id in self.inventory:
            if self.inventory[product_id]['quantity'] >= quantity:
                self.inventory[product_id]['quantity'] -= quantity
                self.log_transaction("REMOVE", product_id, quantity)
                print(f"Removed {quantity} units of {self.inventory[product_id]['name']}")
            else:
                print("Insufficient stock")
        else:
            print("Product not found")
    
    def check_low_stock(self, threshold=5):
        low_stock_items = []
        for pid, product in self.inventory.items():
            if product['quantity'] <= threshold:
                low_stock_items.append((pid, product['name'], product['quantity']))
        return low_stock_items
    
    def generate_report(self):
        total_value = 0
        print("\nInventory Report:")
        print("-" * 60)
        print(f"{'ID':<8} {'Name':<20} {'Qty':<8} {'Price':<10} {'Value':<10}")
        print("-" * 60)
        
        for pid, product in self.inventory.items():
            value = product['quantity'] * product['price']
            total_value += value
            print(f"{pid:<8} {product['name']:<20} {product['quantity']:<8} ${product['price']:<9.2f} ${value:<9.2f}")
        
        print("-" * 60)
        print(f"Total Inventory Value: ${total_value:.2f}")

# Demo inventory system
inventory = InventorySystem()

# Add sample products
sample_products = {
    "P001": {"name": "Laptop", "quantity": 10, "price": 999.99, "supplier": "TechCorp"},
    "P002": {"name": "Mouse", "quantity": 25, "price": 29.99, "supplier": "TechCorp"},
    "P003": {"name": "Keyboard", "quantity": 15, "price": 79.99, "supplier": "InputDevices"},
    "P004": {"name": "Monitor", "quantity": 3, "price": 299.99, "supplier": "DisplayTech"}
}

inventory.inventory = sample_products
inventory.save_inventory()

inventory.generate_report()

# Test stock operations
inventory.add_stock("P001", 5)
inventory.remove_stock("P002", 10)

# Check low stock
low_stock = inventory.check_low_stock(threshold=5)
if low_stock:
    print("\nLow Stock Alert:")
    for pid, name, qty in low_stock:
        print(f"  {pid}: {name} - Only {qty} left!")

inventory.save_inventory()

# Clean up
for file in ["inventory.csv", "transactions.log"]:
    if os.path.exists(file):
        os.remove(file)
print("\nCleaned up inventory files")
print()

print("=== CHALLENGE EXERCISE SOLUTIONS ===\n")

# Challenge 1: Configuration File Manager
print("Challenge 1: Configuration File Manager")
print("-" * 40)

class ConfigManager:
    def __init__(self, config_file="config.json"):
        self.config_file = config_file
        self.config = {}
        self.load_config()
    
    def load_config(self):
        try:
            with open(self.config_file, 'r') as file:
                self.config = json.load(file)
        except FileNotFoundError:
            self.config = self.get_default_config()
            self.save_config()
    
    def get_default_config(self):
        return {
            "database": {
                "host": "localhost",
                "port": 5432,
                "name": "myapp",
                "user": "admin"
            },
            "logging": {
                "level": "INFO",
                "file": "app.log",
                "max_size": "10MB"
            },
            "features": {
                "debug_mode": False,
                "cache_enabled": True,
                "max_connections": 100
            }
        }
    
    def save_config(self):
        # Create backup first
        if os.path.exists(self.config_file):
            backup_file = f"{self.config_file}.backup"
            shutil.copy2(self.config_file, backup_file)
        
        with open(self.config_file, 'w') as file:
            json.dump(self.config, file, indent=2)
        print(f"Configuration saved to {self.config_file}")
    
    def get_setting(self, section, key):
        return self.config.get(section, {}).get(key)
    
    def set_setting(self, section, key, value):
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value
        print(f"Updated {section}.{key} = {value}")
    
    def validate_config(self):
        errors = []
        
        # Validate database settings
        db_config = self.config.get("database", {})
        if not db_config.get("host"):
            errors.append("Database host is required")
        if not isinstance(db_config.get("port"), int):
            errors.append("Database port must be an integer")
        
        # Validate logging settings
        log_config = self.config.get("logging", {})
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if log_config.get("level") not in valid_levels:
            errors.append(f"Logging level must be one of: {valid_levels}")
        
        return errors

# Demo configuration manager
config_mgr = ConfigManager()

print("Current configuration:")
print(json.dumps(config_mgr.config, indent=2))

# Update some settings
config_mgr.set_setting("database", "port", 5433)
config_mgr.set_setting("logging", "level", "DEBUG")
config_mgr.set_setting("features", "debug_mode", True)

# Validate configuration
errors = config_mgr.validate_config()
if errors:
    print("Configuration errors:")
    for error in errors:
        print(f"  - {error}")
else:
    print("Configuration is valid")

config_mgr.save_config()

# Clean up
for file in ["config.json", "config.json.backup"]:
    if os.path.exists(file):
        os.remove(file)
print("Cleaned up configuration files")
print()

print("=== FILE PROCESSING PATTERNS ===\n")

# Large File Processing Example
print("Large File Processing Pattern")
print("-" * 40)

def process_large_csv(filename, chunk_size=1000):
    """Process large CSV files in chunks"""
    processed_count = 0
    
    try:
        with open(filename, 'r') as file:
            reader = csv.DictReader(file)
            
            chunk = []
            for row in reader:
                chunk.append(row)
                
                if len(chunk) >= chunk_size:
                    # Process chunk
                    yield chunk
                    processed_count += len(chunk)
                    chunk = []
                    
                    if processed_count % 10000 == 0:
                        print(f"Processed {processed_count} records...")
            
            # Process remaining records
            if chunk:
                yield chunk
                processed_count += len(chunk)
        
        print(f"Total records processed: {processed_count}")
        
    except FileNotFoundError:
        print(f"File {filename} not found")
    except Exception as e:
        print(f"Error processing file: {e}")

# Create sample large file for demo
large_file = "large_data.csv"
print(f"Creating sample large file: {large_file}")

with open(large_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["ID", "Name", "Value", "Category"])
    
    for i in range(5000):
        writer.writerow([
            i + 1,
            f"Item_{i+1}",
            (i + 1) * 10.5,
            f"Category_{(i % 5) + 1}"
        ])

print("Processing large file in chunks...")
total_value = 0
record_count = 0

for chunk in process_large_csv(large_file, chunk_size=500):
    # Process each chunk
    for record in chunk:
        total_value += float(record['Value'])
        record_count += 1

print(f"Summary: {record_count} records, total value: ${total_value:.2f}")

# Clean up
os.remove(large_file)
print(f"Cleaned up {large_file}")

print("\n" + "=" * 50)
print("File I/O exercise solutions complete!")
