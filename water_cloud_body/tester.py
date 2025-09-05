import subprocess
import sys

def run_test(script_path, expected_output):
    """
    Runs a Python script and checks if its output matches the expected output.
    """
    try:
        # Run the specified Python script and capture its output
        result = subprocess.run(
            ["python", script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )

        # Normalize whitespace for comparison
        actual_output = result.stdout.strip()
        expected_output = expected_output.strip()

        # Compare the script's stdout with the expected output
        if actual_output == expected_output:
            print("OK")
        else:
            print(f"Failed: Expected:\n'{expected_output}'\nGot:\n'{actual_output}'")
    except subprocess.CalledProcessError as e:
        # Handle any exceptions during script execution
        print(f"Failed: Script raised an error - {e.stderr.strip()}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python tester.py <script_to_test.py> <expected_output>")
        print('Example: python tester.py test_script.py "Hello World\\nHow are you?"')
        print("\nNo demo script provided; please provide a script to test.")
        sys.exit(1)

    script_path = sys.argv[1]
    expected_output = sys.argv[2].replace("\\n", "\n")  # Allow newline in expected output via command line
    run_test(script_path, expected_output)