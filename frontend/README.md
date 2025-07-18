# Healthcare Data System Frontend

A clean and intuitive Streamlit frontend for testing the Healthcare Data Generation API.

## Features

- **API Testing Interface**: Test all API endpoints with real-time feedback
- **Data Generation**: Generate synthetic healthcare data with customizable parameters
- **Data Visualization**: Interactive charts and graphs for data analysis
- **System Monitoring**: Real-time health status and performance metrics
- **Modular Design**: Clean, maintainable code structure

## Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the API Server** (in another terminal)
   ```bash
   cd ..
   python scripts/08_run_api.py
   ```

3. **Launch the Frontend**
   ```bash
   streamlit run app.py
   ```

4. **Access the Interface**
   - Open your browser to: http://localhost:8501
   - The frontend will automatically connect to the API at: http://localhost:8000

## Project Structure

```
frontend/
├── app.py                  # Main Streamlit application
├── components/             # Reusable UI components
│   ├── api_client.py      # API communication client
│   └── ui_components.py   # Streamlit UI components
├── utils/                  # Utility functions
│   └── data_processing.py # Data analysis and visualization
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Usage Guide

### 1. API Testing Tab
- Test individual API endpoints
- View request/response details
- Monitor response times and status codes
- Test with custom parameters

### 2. Data Generation Tab
- Generate synthetic healthcare data
- Preview generated datasets
- Download data in multiple formats (CSV, JSON)
- View data quality metrics

### 3. Statistics Tab
- View dataset statistics
- Analyze data distributions
- Monitor system performance

### 4. System Health Tab
- Check API connectivity
- Monitor system components
- View performance metrics
- Access troubleshooting tools

## Configuration

The frontend automatically reads API configuration from the main project's `config/settings.py` file. Default settings:

- **API Host**: localhost
- **API Port**: 8000
- **Max Patients**: 1000
- **Timeout**: 30 seconds

## Troubleshooting

### API Connection Issues
1. Ensure the API server is running: `python scripts/08_run_api.py`
2. Check if the API is accessible: http://localhost:8000/health
3. Verify no firewall is blocking port 8000
4. Check the API logs for errors

### Data Visualization Issues
- Ensure generated data contains expected columns
- Check for missing or null values in key fields
- Verify data types are correct (numeric fields should be numbers)

### Performance Issues
- Reduce the number of patients for generation
- Close unused browser tabs
- Restart the Streamlit application

## Development

### Adding New Features
1. Create new components in the `components/` directory
2. Add utility functions to `utils/`
3. Import and use in `app.py`
4. Follow the existing code style and documentation patterns

### Code Style
- Use type hints for function parameters and returns
- Add docstrings for all functions and classes
- Follow PEP 8 naming conventions
- Keep functions focused and modular

## Dependencies

- **Streamlit**: Web framework for the frontend
- **Requests**: HTTP client for API communication
- **Pandas**: Data manipulation and analysis
- **Plotly**: Interactive data visualizations
- **NumPy**: Numerical computing support