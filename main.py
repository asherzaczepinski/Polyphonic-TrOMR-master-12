""" from flask import Flask, request, jsonify
import subprocess
import os

app = Flask(__name__)

@app.route('/inference', methods=['POST'])
def run_inference():
    data = request.get_json()
    if not data or 'image_path' not in data:
        return jsonify({"error": "Request must contain an 'image_path' field in JSON"}), 400

    # Extract the image path from the request
    image_path = data["image_path"]

    # Validate that the image file exists (optional, depending on your use-case)
    if not os.path.isfile(image_path):
        return jsonify({"error": f"File '{image_path}' not found on server."}), 400

    # Path to the inference script
    inference_script = "./tromr/inference.py"

    # Check if the inference script exists
    if not os.path.isfile(inference_script):
        return jsonify({"error": f"Inference script '{inference_script}' not found."}), 400

    # Construct the command
    command = ["python", inference_script, image_path]

    try:
        # Run the inference script
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Check the return code to see if it executed successfully
        if result.returncode == 0:
            return jsonify({"output": result.stdout}), 200
        else:
            return jsonify({"error": result.stderr}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# For local testing or running on Google Cloud Run / App Engine
if __name__ == "__main__":
    # The host='0.0.0.0' and port=8080 are typical for Cloud Run
    app.run(host='0.0.0.0', port=8080)
 """

import subprocess
import os
import sys
import argparse

def run_inference(debug_mode=False):
    """
    Run the tromr inference on the input.png file with optional debugging
    """
    # Set the image path to always be input.png
    image_path = "input.png"

    # Validate that the image file exists
    if not os.path.isfile(image_path):
        print(f"Error: File '{image_path}' not found.")
        return

    # Path to the inference script
    inference_script = "./tromr/inference.py"

    # Check if the inference script exists
    if not os.path.isfile(inference_script):
        print(f"Error: Inference script '{inference_script}' not found.")
        return

    # Construct the command
    command = ["python", inference_script, image_path]
    
    # Add debugging flags if needed
    if debug_mode:
        command.insert(1, "-m")
        command.insert(2, "pdb")
        
    try:
        # Run the inference script
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Check the return code to see if it executed successfully
        if result.returncode == 0:
            print("Inference completed successfully:")
            print(result.stdout)
        else:
            print("Error during inference:")
            print(result.stderr)
            
            # If we got the specific index dtype error, suggest a fix
            if "scatter(): Expected dtype int64 for index" in result.stderr:
                print("\nPossible fix for the scatter index error:")
                print("The error occurs in tromr/model/decoder.py, line 152.")
                print("You should modify the decoder.py file to ensure indices are converted to int64 type.")
                print("\nEdit decoder.py and find line ~152 with scatter_add_ and insert before it:")
                print("# Convert index tensor to int64")
                print("indices = indices.to(torch.int64)  # Add this line")
                print("filtered_rhythm_logits.scatter_add_(...)  # Original line")
    
    except Exception as e:
        print(f"Exception occurred: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run TrOMR inference on input.png')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode with PDB')
    parser.add_argument('--fix', action='store_true', help='Try to automatically fix the dtype error')
    args = parser.parse_args()
    
    if args.fix:
        try:
            # Path to the decoder.py file
            decoder_path = "./tromr/model/decoder.py"
            
            if os.path.isfile(decoder_path):
                with open(decoder_path, 'r') as file:
                    content = file.read()
                
                # Find the scatter_add_ line and add type conversion before it
                if "scatter_add_" in content:
                    fixed_content = content.replace(
                        "filtered_rhythm_logits.scatter_add_(",
                        "# Convert indices to int64 type to fix scatter_add_ error\n        indices = indices.to(torch.int64)\n        filtered_rhythm_logits.scatter_add_("
                    )
                    
                    # Create backup
                    with open(decoder_path + '.backup', 'w') as file:
                        file.write(content)
                    
                    # Write fixed content
                    with open(decoder_path, 'w') as file:
                        file.write(fixed_content)
                    
                    print(f"Added dtype conversion to {decoder_path}")
                    print(f"Original file backed up to {decoder_path}.backup")
                else:
                    print(f"Could not find scatter_add_ in {decoder_path}")
            else:
                print(f"Could not find decoder.py at {decoder_path}")
        except Exception as e:
            print(f"Error trying to fix the file: {str(e)}")
    

if __name__ == "__main__":
    run_inference(debug_mode=args.debug)
