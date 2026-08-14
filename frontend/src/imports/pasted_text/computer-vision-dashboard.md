Design a modern, professional desktop web application UI for a Computer Vision capstone project titled **“Computer Vision-Based Real-Time Human Emotion Recognition Using Facial Expressions.”**

The system uses a live camera to detect and track multiple people dynamically (N persons) and recognize facial expressions in real time. Each detected person should have a unique Person ID, detected expression, and confidence score.

The system recognizes 7 facial-expression classes: **Happy, Sad, Angry, Fear, Surprise, Disgust, and Neutral.**

Create a professional Computer Vision/AI dashboard with exactly **4 main screens: Dashboard, Live Detection, Analytics, and Session History.**

### 1. Dashboard

Create an overview dashboard containing:

* Project title and system status
* Total Sessions
* Total Faces Analyzed
* Average Confidence
* Most Detected Expression
* Recent Session summary
* Expression Distribution donut chart
* Expression Frequency bar chart
* Quick “Start Live Detection” button
* Recent activity section

Use clean statistic cards with appropriate icons.

### 2. Live Detection

This is the most important screen and should visually dominate the application.

Create a large **Live Camera Feed**.

The camera should demonstrate dynamic multi-person detection and tracking.

Show multiple people inside the camera feed with individual bounding boxes.

Example:

Person 1
Happy — 94%

Person 2
Neutral — 87%

Person 3
Surprise — 91%

Person N
Expression — Confidence %

The system must not visually imply a fixed maximum number of people. It should dynamically support all clearly visible faces according to available camera and processing resources.

Add controls:

* Start Camera
* Stop Camera
* Start Analysis
* End Session

Beside the camera feed, create a **Live Statistics panel** containing:

* Current People Detected
* Happy Count
* Sad Count
* Angry Count
* Fear Count
* Surprise Count
* Disgust Count
* Neutral Count
* Average Confidence
* Current FPS
* Session Duration

Include a small real-time expression distribution chart.

Clearly show that each detected person maintains a **Person ID** while being tracked.

### 3. Analytics

Create a detailed analytics screen for completed and active detection sessions.

At the top provide:

* Session ID
* Date
* Session Duration
* Total People Detected
* Average Confidence
* Dominant Expression

Create visualizations for:

* Expression Distribution donut chart
* Expression Frequency bar chart
* Expression Timeline graph
* Confidence Score graph
* Number of detected people over time

Add an **Individual Person Analysis** section.

Allow selection of a tracked person such as:

Person 1
Person 2
Person 3
...
Person N

After selecting a person, display:

* Person ID
* Dominant Expression
* Average Confidence
* Expression Distribution
* Expression Timeline

Example timeline:

Person 1:
Neutral → Neutral → Happy → Happy → Surprise → Neutral

Person 2:
Happy → Happy → Neutral → Sad → Neutral

Make it clear that these are facial-expression classifications based on visual information.

### 4. Session History

Create a professional table showing previous analysis sessions.

Columns:

* Session ID
* Date
* Start Time
* Duration
* Number of People
* Dominant Expression
* Average Confidence
* View Details

Add search and basic filtering by date or dominant expression.

When “View Details” is selected, show a clean session summary containing:

* Session information
* Expression distribution
* Expression timeline
* Number of people analyzed
* Individual Person IDs
* Confidence statistics

### Navigation

Use a fixed left sidebar containing:

* Dashboard
* Live Detection
* Analytics
* Session History

At the bottom of the sidebar show:

**Computer Vision Capstone Project**

Do not include login, signup, payment, messaging, social media, e-commerce, or unnecessary admin features.

### Design Style

Use a modern **dark AI/Computer Vision dashboard theme**.

The interface should feel like a real AI monitoring and analytics application rather than a normal website.

Use:

* Professional dark background
* High-contrast cards
* Modern typography
* Rounded cards
* Subtle shadows
* Minimal gradients
* Clean line icons
* Clear chart designs
* Consistent spacing
* Professional data visualization
* Smooth hover states
* Status indicators

Use suitable icons for camera, face detection, analytics, history, FPS, confidence, timer, and emotions.

Design primarily for a **1440px desktop screen**.

The **Live Detection screen should receive the highest visual priority**, followed by Analytics.

Maintain a consistent design system across all four screens and create reusable components for navigation, buttons, statistic cards, charts, camera controls, Person ID labels, expression labels, and confidence indicators.

The final design should look polished enough for a **4-member B.Tech Computer Vision capstone project presentation and live demonstration**.
