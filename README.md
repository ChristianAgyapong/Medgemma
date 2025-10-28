# 🏥 GEMAA - Google MedGemma Local Environment# MedGemma Local Environment



A complete local setup for working with **Google MedGemma 4B** AI model for medical question answering and analysis.This project sets up a Google Colab-like environment for working with MedGemma (medical language models based on Google's Gemma) locally.



---## 🚀 Quick Start



## 📁 Project Structure### 1. Install Dependencies

```powershell

```.venv\Scripts\python.exe -m pip install --upgrade pip

GEMAA/.venv\Scripts\python.exe -m pip install -r requirements.txt

│```

├── 📓 notebooks/              # Jupyter notebooks

│   ├── medgemma_actual.ipynb       # Main MedGemma 4B notebook (✅ Token configured)### 2. Launch Jupyter Notebook

│   ├── real_world_datasets.ipynb  # Real medical datasets analysis```powershell

│   ├── start_here.ipynb            # Sample data analysis.venv\Scripts\jupyter notebook

│   └── test_cells.ipynb            # Environment testing```

│Or for JupyterLab:

├── 🔧 scripts/                # Python & PowerShell scripts```powershell

│   ├── launch_medgemma.ps1         # 🚀 One-click MedGemma launcher.venv\Scripts\jupyter lab

│   ├── launch_jupyter.ps1          # Launch Jupyter Notebook```

│   ├── diagnose_jupyter.py         # Diagnose setup issues

│   ├── test_setup.py               # Test environment### 3. Open the Setup Notebook

│   └── verify_setup.py             # Verify dependenciesOpen `setup_medgemma.ipynb` and follow the step-by-step instructions.

│

├── 📚 docs/                   # Documentation## 📋 Prerequisites

│   ├── START_HERE.md              # ⭐ Quick start guide

│   ├── QUICK_REFERENCE.md         # Quick lookup reference- **Hugging Face Account**: Create an account at [huggingface.co](https://huggingface.co)

│   ├── HOW_TO_RUN_CELLS.md        # Detailed instructions- **Gemma Access**: Accept the license at [google/gemma-2b](https://huggingface.co/google/gemma-2b)

│   ├── TROUBLESHOOTING.md         # Common issues & fixes- **HF Token**: Generate a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

│   └── MEDGEMMA_4B_GUIDE.md       # MedGemma 4B info

│## 🔧 What's Included

├── 📊 data/                   # Downloaded datasets

├── 📁 outputs/                # Generated outputs### Dependencies

├── 🐍 .venv/                  # Python virtual environment- **PyTorch & TensorFlow**: Deep learning frameworks

├── 📄 requirements.txt        # Python dependencies- **Transformers**: Hugging Face library for LLMs

└── 📄 .gitignore             # Git ignore rules- **Jupyter**: Interactive notebook environment

```- **Medical NLP tools**: Specialized libraries for medical text processing

- **Visualization**: Matplotlib, Seaborn, Plotly

---

### Features

## ✅ What's Installed- 4-bit quantization for efficient memory usage

- Interactive chat interface

- **Python 3.9.13** in virtual environment- Batch processing capabilities

- **PyTorch 2.8.0+cpu** - Deep learning framework- Medical query examples

- **Transformers 4.57.1** - Hugging Face library- Model performance monitoring

- **BitsAndBytes 0.48.1** - 4-bit quantization

- **Jupyter Notebook** - Interactive environment## 💻 Hardware Requirements

- **Data Science Stack** - NumPy, Pandas, Matplotlib, Seaborn

- **MedGemma 4B** - Medical-specific AI (4 billion parameters)### Minimum

- **RAM**: 8GB

---- **Storage**: 10GB free space

- **CPU**: Multi-core processor

## 🚀 Quick Start (3 Steps)

### Recommended

### **Step 1: Accept MedGemma License** (One-time!)- **GPU**: NVIDIA GPU with 8GB+ VRAM (for faster inference)

👉 **https://huggingface.co/google/medgemma-4b-it**  - **RAM**: 16GB+

Click: **"Agree and access repository"**- **Storage**: 20GB+ free space



### **Step 2: Launch MedGemma**## 📚 Model Variants

```powershell

.\scripts\launch_medgemma.ps1You can work with different Gemma models:

```- `google/gemma-2b` - 2 billion parameters (lighter)

- `google/gemma-7b` - 7 billion parameters (more capable)

### **Step 3: Run Cells**- `google/gemma-2b-it` - Instruction-tuned variant

1. Browser opens with notebook

2. **Top right** → Select: `Python 3.9 (GEMAA)`## 🔒 Security Note

3. Run cells: `Shift + Enter`

4. Ask medical questions! 🏥**Never commit your Hugging Face token to version control!**



---Create a `.env` file (already in .gitignore) to store sensitive data:

```

## 📓 NotebooksHF_TOKEN=your_token_here

```

| Notebook | Purpose |

|----------|---------|## 📖 Usage Examples

| **`medgemma_actual.ipynb`** | Main MedGemma 4B AI (token configured ✅) |

| **`real_world_datasets.ipynb`** | Analyze real medical data |### Basic Text Generation

| **`start_here.ipynb`** | Sample data analysis |```python

| **`test_cells.ipynb`** | Test environment |from transformers import AutoTokenizer, AutoModelForCausalLM



---tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")

model = AutoModelForCausalLM.from_pretrained("google/gemma-2b")

## 🔧 Scripts

prompt = "What are the symptoms of diabetes?"

| Command | Purpose |inputs = tokenizer(prompt, return_tensors="pt")

|---------|---------|outputs = model.generate(**inputs, max_new_tokens=100)

| `.\scripts\launch_medgemma.ps1` | 🚀 Launch MedGemma notebook |response = tokenizer.decode(outputs[0])

| `.\scripts\launch_jupyter.ps1` | Launch Jupyter file browser |print(response)

| `.venv\Scripts\python.exe scripts\diagnose_jupyter.py` | Diagnose issues |```



---### Medical Question Answering

See `setup_medgemma.ipynb` for comprehensive examples of:

## 📚 Documentation- Interactive chat

- Batch processing

All guides in `docs/` folder:- Fine-tuning preparation

- **`START_HERE.md`** - Complete setup guide ⭐- Performance monitoring

- **`QUICK_REFERENCE.md`** - Fast lookup

- **`HOW_TO_RUN_CELLS.md`** - Cell execution guide## 🛠️ Troubleshooting

- **`TROUBLESHOOTING.md`** - Problem solutions

### Out of Memory Error

---- Use 4-bit or 8-bit quantization

- Try a smaller model (gemma-2b instead of gemma-7b)

## 🎯 Common Tasks- Reduce batch size



### Ask Medical Questions### Model Download Issues

```powershell- Check your internet connection

.\scripts\launch_medgemma.ps1- Verify Hugging Face token is valid

```- Ensure you've accepted the Gemma license



### Analyze Real Medical Data### CUDA Not Available

```powershell- Install CUDA toolkit if you have an NVIDIA GPU

.\scripts\launch_jupyter.ps1- Update GPU drivers

# Then open: notebooks/real_world_datasets.ipynb- The model will fall back to CPU (slower but functional)

```

## 📝 License

### Test Setup

```powershellThis project uses Google's Gemma models, which require accepting their license agreement.

.venv\Scripts\python.exe scripts\diagnose_jupyter.py

```## 🤝 Contributing



---Feel free to open issues or submit pull requests to improve this setup!



## ⚠️ Important## 📞 Support



### Before First Use:For issues with:

1. ✅ Accept license: https://huggingface.co/google/medgemma-4b-it- Gemma models: Visit [Hugging Face Gemma page](https://huggingface.co/google/gemma-2b)

2. ✅ Select kernel: `Python 3.9 (GEMAA)`- This setup: Open an issue in this repository

3. ⏳ First run: Cell 3 downloads 2GB model (5-10 min)

---

### Token Configured:

- ✅ Already in `notebooks/medgemma_actual.ipynb`**Happy Medical AI Development! 🏥🤖**

- ✅ No manual login needed
- ⚠️ Don't commit to public repos

---

## 🆘 Troubleshooting

**Cells don't run?**  
→ Select kernel: `Python 3.9 (GEMAA)` (top right)

**"No module named 'torch'"?**  
→ Wrong kernel selected

**Cell shows `[*]` forever?**  
→ If Cell 3: Normal (downloading model)

See `docs/TROUBLESHOOTING.md` for more help.

---

## 🎓 Disclaimer

**For educational purposes only.**  
Always consult healthcare professionals for medical advice.

---

## 🎉 You're Ready!

1. Accept license: https://huggingface.co/google/medgemma-4b-it
2. Run: `.\scripts\launch_medgemma.ps1`
3. Select kernel: `Python 3.9 (GEMAA)`
4. Ask questions! 🏥

**Happy Medical AI Exploration!** 🚀
