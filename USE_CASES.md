# Forensic Video RAG System - Use Cases

Simple and effective real-world scenarios where this system provides value.

---

## Use Case 1: Package Theft Investigation

**Scenario**: A package was stolen from a lobby. Security needs to find who took it.

**Query Examples**:

- "Show me who stole the package from the lobby"
- "Find the person who took the package at 3:45 PM"
- "Show me all people near the package delivery area between 3:00-4:00 PM"

**What System Does**:

1. Detects person + package interactions in video
2. Identifies the suspect's face
3. Shows timeline: when they entered, approached package, left
4. Provides evidence frames with timestamps

**Value**: Saves hours of manual video review

---

## Use Case 2: Unauthorized Access Detection

**Scenario**: Someone entered a restricted area after hours.

**Query Examples**:

- "Who entered the building after 9 PM?"
- "Show me all people in the server room"
- "Find unauthorized access to the storage room"

**What System Does**:

1. Tracks all person detections with timestamps
2. Identifies faces of people in restricted areas
3. Creates timeline of entry/exit
4. Flags suspicious behavior (after-hours access)

**Value**: Quick identification of security breaches

---

## Use Case 3: Person Re-Identification Across Cameras

**Scenario**: A suspect was seen in lobby camera, need to find them in parking lot camera.

**Query Examples**:

- "Find the person in the red hoodie from the lobby in other cameras"
- "Show me where Person_02 appears in all videos"
- "Track the suspect across all camera feeds"

**What System Does**:

1. Uses face recognition to match person across different videos
2. Shows all appearances of the same person
3. Creates a path/timeline across multiple cameras
4. Displays face matches with confidence scores

**Value**: Connects events across multiple camera feeds automatically

---

## Use Case 4: Timeline Reconstruction

**Scenario**: Need to understand the sequence of events leading to an incident.

**Query Examples**:

- "What happened before the theft?"
- "Show me the timeline of events at 2:30 PM"
- "What did Person_05 do between 10:00-11:00 AM?"

**What System Does**:

1. Uses Neo4j graph to show event sequence
2. Displays chronological order of scenes
3. Shows relationships: Person X was in Scene A, then Scene B
4. Visualizes the timeline as a graph

**Value**: Understands context and sequence of events

---

## Use Case 5: Loitering Detection

**Scenario**: Security wants to find people who stayed in an area too long.

**Query Examples**:

- "Find people who stayed in the lobby for more than 10 minutes"
- "Show me loitering behavior near the entrance"
- "Who was in the parking lot for extended time?"

**What System Does**:

1. Tracks person positions over time
2. Calculates duration in specific areas
3. Flags suspicious loitering patterns
4. Shows person's path and time spent

**Value**: Identifies potential security concerns automatically

---

## Use Case 6: Vehicle Tracking

**Scenario**: Need to find a specific vehicle or track vehicle movements.

**Query Examples**:

- "Show me all cars that entered the parking lot"
- "Find the red car that left at 5:30 PM"
- "Track vehicle movements in the parking area"

**What System Does**:

1. Detects and tracks vehicles (cars, trucks, motorcycles)
2. Records entry/exit times
3. Tracks vehicle paths
4. Can match vehicles across different camera angles

**Value**: Vehicle-based investigations and access control

---

## Use Case 7: Evidence Compilation

**Scenario**: Need to create a report with all evidence related to an incident.

**Query Examples**:

- "Compile all evidence for the incident at 3:45 PM"
- "Show me all scenes involving Person_02"
- "Create evidence report for the theft case"

**What System Does**:

1. Retrieves all relevant keyframes
2. Shows face matches and timestamps
3. Creates a visual timeline
4. Exports evidence package (frames + metadata)

**Value**: Quick evidence compilation for reports or legal use

---

## Use Case 8: Behavior Pattern Analysis

**Scenario**: Security wants to understand normal vs. suspicious behavior patterns.

**Query Examples**:

- "Show me unusual behavior in the lobby"
- "Find people who entered but didn't leave normally"
- "What activities happened during the incident?"

**What System Does**:

1. Uses Gemini VLM to analyze scene semantics
2. Identifies suspicious activities (theft, violence, loitering)
3. Compares against normal patterns
4. Flags anomalies

**Value**: Proactive security monitoring and anomaly detection

---

## Use Case 9: Quick Person Search

**Scenario**: Need to find a specific person quickly in hours of footage.

**Query Examples**:

- "Find the person wearing a blue jacket"
- "Show me all appearances of the person with glasses"
- "Search for people with backpacks"

**What System Does**:

1. Uses semantic search (ChromaDB) to find descriptions
2. Matches visual descriptions from Gemini analysis
3. Shows all matching scenes with timestamps
4. Allows filtering by time, location, appearance

**Value**: Finds specific people in minutes instead of hours

---

## Use Case 10: Multi-Camera Investigation

**Scenario**: Incident spans multiple camera feeds, need unified view.

**Query Examples**:

- "Show me Person_03 across all cameras"
- "What happened in the lobby and parking lot simultaneously?"
- "Track the suspect through all camera views"

**What System Does**:

1. Correlates events across multiple video files
2. Shows unified timeline from all cameras
3. Tracks person movement across different views
4. Creates comprehensive investigation view

**Value**: Single unified view of multi-camera incidents

---

## Summary: Key Benefits

1. **Speed**: Find answers in minutes vs. hours of manual review
2. **Accuracy**: Face recognition + semantic search reduces human error
3. **Evidence Trail**: Verifiable timestamps and frame references
4. **Natural Language**: Ask questions in plain English
5. **Cost-Effective**: Uses free-tier cloud APIs + local processing
6. **Privacy**: Video never leaves your device (only keyframes to cloud)

---

## Target Users

- **Security Teams**: Quick incident investigation
- **Law Enforcement**: Evidence gathering and timeline reconstruction
- **Property Managers**: Access control and unauthorized entry detection
- **Forensic Analysts**: Detailed investigation with evidence trails
