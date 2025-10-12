# Lesson 00: Git and GitHub Setup

## Learning Objectives
By the end of this lesson, you will be able to:
- Understand what Git and GitHub are and why they're important
- Install Git on Windows 11
- Create a GitHub account and repository
- Use basic Git commands to manage code
- Set up your personal repository for submitting exercises

## What is Git and GitHub?

### Git
Git is a **version control system** - think of it as a sophisticated "save" system for your code that:
- Tracks every change you make to your files
- Allows you to go back to previous versions
- Helps multiple people work on the same project
- Creates a complete history of your project

### GitHub
GitHub is a **cloud-based platform** that:
- Stores your Git repositories online
- Allows you to share code with others
- Provides backup for your projects
- Enables collaboration with other developers

### Why Do We Need This?
As you learn programming, you'll want to:
- Save different versions of your work
- Share your progress with your instructor
- Build a portfolio of your projects
- Learn industry-standard tools

## Installing Git on Windows 11

### Step 1: Download Git
1. Go to [https://git-scm.com/download/win](https://git-scm.com/download/win)
2. The download should start automatically
3. If not, click "64-bit Git for Windows Setup"

### Step 2: Install Git
1. Run the downloaded installer
2. **Important settings during installation:**
   - **Select Components**: Keep all default selections
   - **Default editor**: Choose "Use Visual Studio Code" if you have it, otherwise "Use Notepad"
   - **PATH environment**: Choose "Git from the command line and also from 3rd-party software"
   - **Line ending conversions**: Choose "Checkout Windows-style, commit Unix-style line endings"
   - **Terminal emulator**: Choose "Use Windows' default console window"
   - Keep all other default settings

### Step 3: Verify Installation
1. Open Command Prompt (press `Win + R`, type `cmd`, press Enter)
2. Type: `git --version`
3. You should see something like: `git version 2.42.0.windows.1`

## Setting Up GitHub

### Step 1: Create GitHub Account
1. Go to [https://github.com](https://github.com)
2. Click "Sign up"
3. Choose a username (this will be public - consider using your name or a professional variation)
4. Use your email address
5. Create a strong password
6. Verify your account through email

### Step 2: Configure Git with Your Information
Open Command Prompt and run these commands (replace with your actual information):

```bash
git config --global user.name "Your Full Name"
git config --global user.email "your.email@example.com"
```

### Step 3: Create Your First Repository
1. On GitHub, click the "+" icon in the top right
2. Select "New repository"
3. Repository name: `python-learning-exercises`
4. Description: "My Python learning journey and exercise solutions"
5. Make it **Public** (so your instructor can see it)
6. Check "Add a README file"
7. Click "Create repository"

## Basic Git Commands

### Essential Commands You'll Use

```bash
# Clone a repository (download it to your computer)
git clone https://github.com/username/repository-name.git

# Check the status of your files
git status

# Add files to be committed (staged)
git add filename.py          # Add specific file
git add .                    # Add all changed files

# Commit your changes (save a snapshot)
git commit -m "Describe what you changed"

# Push changes to GitHub (upload)
git push

# Pull latest changes (download updates)
git pull
```

### Your Workflow
This is what you'll do for each exercise:

1. **Make changes** to your code
2. **Check status**: `git status`
3. **Add files**: `git add .`
4. **Commit**: `git commit -m "Completed lesson X exercises"`
5. **Push**: `git push`

## Setting Up Your Exercise Repository

### Step 1: Clone the Course Repository
1. Open Command Prompt
2. Navigate to where you want to store your code:
   ```bash
   cd C:\Users\YourUsername\Documents
   mkdir PythonLearning
   cd PythonLearning
   ```
3. Clone this course repository:
   ```bash
   git clone [INSTRUCTOR_WILL_PROVIDE_URL]
   ```

### Step 2: Clone Your Personal Repository
```bash
git clone https://github.com/yourusername/python-learning-exercises.git
cd python-learning-exercises
```

### Step 3: Create Folder Structure
```bash
mkdir lesson-01
mkdir lesson-02
# ... we'll create more as we go
```

### Step 4: Test Your Setup
1. Create a test file:
   ```bash
   echo "# My Python Learning Journey" > test.md
   ```
2. Add and commit:
   ```bash
   git add test.md
   git commit -m "Initial test commit"
   git push
   ```
3. Check GitHub - you should see your file there!

## Common Git Scenarios

### When Things Go Wrong
- **Forgot to commit before making new changes**: Commit current changes first
- **Made a mistake in commit message**: Use `git commit --amend -m "New message"`
- **Want to see what changed**: Use `git diff`
- **Want to see commit history**: Use `git log --oneline`

### Best Practices
- **Commit often**: After completing each exercise or making significant changes
- **Write clear commit messages**: "Completed exercise 3" not "stuff"
- **Pull before you push**: Always `git pull` before `git push` if working with others
- **Don't commit sensitive information**: Never commit passwords or personal data

## Understanding the Workflow

```
Your Computer          GitHub
     |                    |
     |  git clone         |
     |<-------------------|
     |                    |
     |  (make changes)    |
     |                    |
     |  git add .         |
     |  git commit        |
     |                    |
     |  git push          |
     |-------------------->
     |                    |
```

## Troubleshooting Common Issues

### "Permission denied" when pushing
- You may need to set up authentication
- GitHub now requires personal access tokens instead of passwords
- Your instructor will help you set this up if needed

### "Repository not found"
- Check the URL is correct
- Make sure the repository is public or you have access

### "Nothing to commit"
- You haven't made any changes
- Or you forgot to `git add` your files

## Next Steps

In our next lesson, we'll start programming! You'll use your Git skills to:
- Save your first Python programs
- Submit exercises to your repository
- Track your progress as you learn

Remember: Git might feel complicated at first, but these basic commands will become second nature with practice. Don't worry about understanding everything immediately - we'll reinforce these concepts throughout the course.
