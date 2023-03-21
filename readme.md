# Filtering and edge detection

A web application that use python script for image processing such as detect edges using canny detection , Detect lines , Detect circles , Detect ellipse and initialize the contour and evolve the Active Contour Model (snake) using the greedy algorithm.

> ## Table of Contents

- [Built with](#Built-with)
- [Deployment](#Deployment)
- [Design](#Design)
- [Features](#Features)
- [Authors](#Authors)


> ## Built with

![programming language](https://img.shields.io/badge/programmig%20language-Python-red)
![Framework](https://img.shields.io/badge/Framework-Streamlit-blue)
![styling](https://img.shields.io/badge/Styling-CSS-ff69b4)


> ## Deployment

 Install streamlit

```bash
  pip install streamlit
```

To start deployment 
```bash
  streamlit run app.py
```

> ## 🖌️ Design

![main widow](./Demo/Home.png)


> ## Features
###  Detect edges using Canny detection

![main widow](./Demo/canny.gif)


#### Detect objects using Hough transform
1. Detect lines
- Default parameter
![main widow](./Demo/line1.png)

- Effect of Low threshold and High threshold parameter
![main widow](./Demo/line2.png)

- Effect of Neighboorhood size parameter
![main widow](./Demo/line3.png)



2. Detect ellipse
- Default parameter
![main widow](./Demo/ellipse1.png)

- Effect of Thickness parameter
![main widow](./Demo/ellipse2.png)

3. Detect circles


###  Apply Active Contour (Snake)



> ## 🔗 Authors
- Esraa Ali         
sec : 1   BN : 12

- Rawan Abdulhamid  
sec : 1   BN : 33

- Mostafa Mahmoud   
sec : 2   BN : 37

- Omar Mustafa      
sec : 2   BN : 5  

- Yehia Said        
sec : 2   BN : 53 


All rights reserved © 2023 to Team 9 - Systems & Biomedical Engineering, Cairo University (Class 2024)