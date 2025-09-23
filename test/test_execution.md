tests can be executed as follows
# Run all tests
python -m unittest discover -s test -p "test_*.py"

# Run a specific test file
python -m unittest test.game.test_arena

# Run with verbose output
python -m unittest discover -s test -p "test_*.py" -v
