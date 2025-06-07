# sd2-attention-visualization

This project provides a framework for visualizing attention maps in the Stable Diffusion 2.0 model using the `diffusers` library. It allows users to generate images based on input prompts while capturing and visualizing the attention mechanisms of the model.

## Project Structure

```
sd2-attention-visualization
├── src
│   ├── main.py                # Entry point for generating images and visualizing attention
│   ├── attention_visualizer.py # Functions for visualizing attention maps
│   ├── attn_processor.py       # Modified attention processor to capture attention maps
│   └── utils.py                # Utility functions for image processing
├── requirements.txt            # List of dependencies
└── README.md                   # Project documentation
```

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd sd2-attention-visualization
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run the application:**
   You can generate images and visualize attention maps by running the main script:
   ```bash
   python src/main.py --prompt "A fantasy landscape" --num_images 5
   ```

2. **Parameters:**
   - `--prompt`: The text prompt for image generation.
   - `--num_images`: The number of images to generate.

## Attention Visualization Process

The modified `AttnProcessor` class captures attention maps during the image generation process. These maps can be visualized using the functions provided in `attention_visualizer.py`. The attention maps are overlaid on the generated images to provide insights into the model's focus areas during generation.

## Example

After running the application, generated images along with their corresponding attention maps will be saved in the output directory. You can view the attention maps to understand how the model interprets different parts of the input prompt.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.