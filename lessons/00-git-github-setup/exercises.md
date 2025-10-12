# Lesson 00 Exercises: Git and GitHub Setup

## Guided Exercises (Do with Instructor)

### Exercise 1: Installation Verification
**Goal**: Confirm Git is properly installed and configured

**Steps**:
1. Open Command Prompt
2. Check Git version: `git --version`
3. Check your configuration:
   ```bash
   git config --global user.name
   git config --global user.email
   ```
4. If not set, configure with your information

**Expected Output**: You should see your name and email displayed

---

### Exercise 2: Create and Setup Your Repository
**Goal**: Create your personal learning repository

**Steps**:
1. Go to GitHub and create a new repository named `python-learning-exercises`
2. Make it public and add a README
3. Clone it to your computer in a `PythonLearning` folder
4. Navigate into the cloned repository folder

**Verification**: You should have a local folder with a README.md file

---

### Exercise 3: Your First Commit
**Goal**: Practice the basic Git workflow

**Steps**:
1. In your repository folder, create a file called `about-me.md`
2. Add this content (replace with your information):
   ```markdown
   # About Me
   
   **Name**: [Your Name]
   **Goal**: Learn Python for data analysis
   **Experience**: Complete beginner
   **Date Started**: [Today's Date]
   ```
3. Save the file
4. Add it to Git: `git add about-me.md`
5. Commit it: `git commit -m "Add about me file"`
6. Push to GitHub: `git push`

**Verification**: Check GitHub - your file should appear there

---

## Independent Exercises (Try on Your Own)

### Exercise 4: Create Lesson Folders
**Goal**: Set up organization for future lessons

**Tasks**:
1. Create folders for the first 5 lessons:
   - `lesson-01-intro-programming`
   - `lesson-02-python-basics`
   - `lesson-03-control-flow`
   - `lesson-04-functions`
   - `lesson-05-data-structures`

2. In each folder, create an empty file called `notes.md`

3. Add, commit, and push all changes with the message "Set up lesson folders"

**Hint**: You can create folders with `mkdir foldername` and files with `echo "# Lesson Notes" > filename.md`

---

### Exercise 5: Practice Git Status and Diff
**Goal**: Learn to check what's changed before committing

**Tasks**:
1. Edit your `about-me.md` file to add a new line:
   ```markdown
   **Favorite Programming Concept So Far**: Version control with Git!
   ```

2. Before adding/committing, run:
   - `git status` (see what files changed)
   - `git diff` (see exactly what changed)

3. Add, commit, and push the changes

**Learning Point**: Always check `git status` before committing to see what you're about to save

---

### Exercise 6: Explore Git History
**Goal**: Learn to view your project history

**Tasks**:
1. Run `git log --oneline` to see your commit history
2. Run `git log --graph --oneline` for a visual representation
3. Create a file called `git-commands.md` and list the 5 most important Git commands you've learned
4. Commit and push this file

**Reflection**: Write in your notes which Git command you think will be most useful

---

## Challenge Exercises (If You Finish Early)

### Challenge 1: Markdown Practice
**Goal**: Get comfortable with Markdown formatting

**Task**: Create a file called `learning-goals.md` with:
- A main heading
- At least 3 subheadings
- A bulleted list of what you want to learn
- A numbered list of your learning priorities
- Some **bold** and *italic* text
- A link to a programming resource you found online

### Challenge 2: Repository Exploration
**Goal**: Explore other repositories to see real-world examples

**Tasks**:
1. Visit some popular Python repositories on GitHub:
   - https://github.com/python/cpython
   - https://github.com/pandas-dev/pandas
   - https://github.com/matplotlib/matplotlib

2. Look at their README files and folder structures
3. Create a file called `repository-observations.md` with notes about what you noticed

### Challenge 3: Git Branching (Advanced)
**Goal**: Learn about Git branches (we'll cover this more later)

**Tasks**:
1. Create a new branch: `git checkout -b experiment`
2. Make some changes to any file
3. Commit the changes
4. Switch back to main: `git checkout main`
5. Notice your changes aren't there anymore
6. Switch back to experiment: `git checkout experiment`
7. See your changes return!

**Note**: Don't worry if this seems confusing - we'll cover branching in detail later

---

## Troubleshooting Guide

### Common Issues and Solutions

**Problem**: `git: command not found`
- **Solution**: Git isn't installed or not in PATH. Reinstall Git and restart Command Prompt

**Problem**: `Permission denied (publickey)`
- **Solution**: You need to set up authentication. Ask instructor for help with personal access tokens

**Problem**: `fatal: not a git repository`
- **Solution**: You're not in a Git repository folder. Use `cd` to navigate to your repository

**Problem**: `nothing to commit, working tree clean`
- **Solution**: You haven't made any changes, or you forgot to save your files

**Problem**: Files not showing up on GitHub after push
- **Solution**: Make sure you used `git add` before `git commit`

---

## Self-Assessment Checklist

After completing these exercises, you should be able to:

- [ ] Install and configure Git on Windows 11
- [ ] Create a GitHub repository
- [ ] Clone a repository to your computer
- [ ] Use basic Git commands (add, commit, push, status)
- [ ] Create and edit files in your repository
- [ ] View your commit history
- [ ] Understand the basic Git workflow

If you can't check all these boxes, review the lesson material or ask your instructor for help before moving to the next lesson.
