Road Network Optimization using AI
___________________________________

This project focuses on AI-driven road network optimization using Django and Genetic Algorithms (GA). It extracts road networks using OpenStreetMap (OSM) via OSMnx, then optimizes road placement by avoiding obstacles like buildings while maximizing efficiency. A Django-based API is built to generate and optimize road layouts, providing results in the form of JSON responses with base64-encoded images. The project includes visualization capabilities through Matplotlib, enabling users to compare original and optimized road networks.

To set up the project, clone the repository, install the dependencies, and start the Django server. The /generate_road_network API endpoint allows users to optimize road layouts for a specified region, while a default view provides a welcome message. The repository follows a structured project layout, including Python scripts for AI logic, Django templates, and static files for visualization. Future enhancements may include deep learning-based optimization, an interactive UI, and cloud deployment for broader accessibility.
