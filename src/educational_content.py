"""
Educational Content Manager

Manages educational content, learning paths, progress tracking, and insights
for the Gravity Yonder Over physics education platform.
"""

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import pandas as pd
import numpy as np


@dataclass
class LearningObjective:
    """Learning objective definition"""
    id: str
    title: str
    description: str
    physics_concepts: List[str]
    difficulty_level: int  # 1-5
    prerequisites: List[str]
    simulation_scenarios: List[str]


@dataclass
class UserProgress:
    """User progress tracking"""
    user_id: str
    lesson_id: str
    completion_percentage: float
    time_spent: float  # minutes
    last_accessed: str
    quiz_scores: List[float]
    simulation_interactions: int
    mastery_level: float  # 0-1


@dataclass
class QuizQuestion:
    """Quiz question structure"""
    id: str
    question: str
    question_type: str  # 'multiple_choice', 'numerical', 'conceptual'
    options: List[str]  # For multiple choice
    correct_answer: str
    explanation: str
    physics_concept: str
    difficulty: int


class EducationalContentManager:
    """
    Manages educational content and user progress for the physics platform.
    
    Features:
    - Learning path management
    - Progress tracking
    - Adaptive content delivery
    - Quiz generation and assessment
    - Insight generation
    - Performance analytics
    """
    
    def __init__(self, content_dir: str = "curriculum", db_path: str = "data/user_progress.db"):
        self.content_dir = Path(content_dir)
        self.db_path = Path(db_path)
        
        # Ensure directories exist
        self.content_dir.mkdir(exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Load educational content
        self._load_learning_objectives()
        self._load_quiz_questions()
        
        # Initialize physics concept mapping
        self._init_physics_concepts()
    
    def _init_database(self):
        """Initialize SQLite database for progress tracking"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # User progress table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_progress (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                lesson_id TEXT NOT NULL,
                completion_percentage REAL,
                time_spent REAL,
                last_accessed TEXT,
                quiz_scores TEXT,
                simulation_interactions INTEGER,
                mastery_level REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, lesson_id)
            )
        ''')
        
        # Learning analytics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS learning_analytics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                event_data TEXT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                session_id TEXT
            )
        ''')
        
        # Quiz attempts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS quiz_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                quiz_id TEXT NOT NULL,
                question_id TEXT NOT NULL,
                user_answer TEXT,
                correct_answer TEXT,
                is_correct BOOLEAN,
                response_time REAL,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_learning_objectives(self):
        """Load learning objectives from curriculum files"""
        self.learning_objectives = {
            "gravity_basics": LearningObjective(
                id="gravity_basics",
                title="Introduction to Gravity",
                description="Understand fundamental concepts of gravitational force and fields",
                physics_concepts=["inverse_square_law", "gravitational_field", "potential_energy"],
                difficulty_level=1,
                prerequisites=[],
                simulation_scenarios=["binary_orbit"]
            ),
            "orbital_mechanics": LearningObjective(
                id="orbital_mechanics",
                title="Orbital Mechanics",
                description="Learn about planetary motion and orbital dynamics",
                physics_concepts=["keplers_laws", "orbital_energy", "angular_momentum"],
                difficulty_level=2,
                prerequisites=["gravity_basics"],
                simulation_scenarios=["planetary_system", "binary_orbit"]
            ),
            "black_holes": LearningObjective(
                id="black_holes",
                title="Black Holes and Extreme Gravity",
                description="Explore black holes and relativistic effects",
                physics_concepts=["event_horizon", "spacetime_curvature", "tidal_forces"],
                difficulty_level=4,
                prerequisites=["gravity_basics", "orbital_mechanics"],
                simulation_scenarios=["black_hole_accretion"]
            ),
            "gravitational_waves": LearningObjective(
                id="gravitational_waves",
                title="Gravitational Waves",
                description="Understanding ripples in spacetime",
                physics_concepts=["general_relativity", "wave_propagation", "binary_mergers"],
                difficulty_level=5,
                prerequisites=["black_holes"],
                simulation_scenarios=["gravitational_waves"]
            )
        }
    
    def _load_quiz_questions(self):
        """Load quiz questions for different topics"""
        self.quiz_questions = {
            "gravity_basics": [
                QuizQuestion(
                    id="gb_01",
                    question="What is the relationship between gravitational force and distance?",
                    question_type="multiple_choice",
                    options=[
                        "Force increases with distance",
                        "Force decreases as 1/r²",
                        "Force decreases as 1/r",
                        "Force is independent of distance"
                    ],
                    correct_answer="Force decreases as 1/r²",
                    explanation="Newton's law of universal gravitation states F = G·m₁·m₂/r², so force decreases as the inverse square of distance.",
                    physics_concept="inverse_square_law",
                    difficulty=1
                ),
                QuizQuestion(
                    id="gb_02",
                    question="If Earth's mass doubled, what would happen to your weight?",
                    question_type="multiple_choice",
                    options=[
                        "Stay the same",
                        "Double",
                        "Halve",
                        "Quadruple"
                    ],
                    correct_answer="Double",
                    explanation="Weight = mg, where g depends on Earth's mass. Doubling Earth's mass doubles g, thus doubling weight.",
                    physics_concept="gravitational_field",
                    difficulty=2
                ),
                QuizQuestion(
                    id="gb_03",
                    question="What is the escape velocity from Earth's surface (approximately)?",
                    question_type="multiple_choice",
                    options=[
                        "7.9 km/s",
                        "11.2 km/s",
                        "25 km/s",
                        "42 km/s"
                    ],
                    correct_answer="11.2 km/s",
                    explanation="Escape velocity v = √(2GM/r) ≈ 11.2 km/s for Earth.",
                    physics_concept="escape_velocity",
                    difficulty=3
                )
            ],
            "orbital_mechanics": [
                QuizQuestion(
                    id="om_01",
                    question="According to Kepler's Third Law, if a planet's orbital radius doubles, its period:",
                    question_type="multiple_choice",
                    options=[
                        "Doubles",
                        "Quadruples",
                        "Increases by 2√2",
                        "Stays the same"
                    ],
                    correct_answer="Increases by 2√2",
                    explanation="Kepler's 3rd law: T² ∝ r³, so T ∝ r^(3/2). If r doubles, T increases by 2^(3/2) = 2√2.",
                    physics_concept="keplers_laws",
                    difficulty=3
                ),
                QuizQuestion(
                    id="om_02",
                    question="In an elliptical orbit, where is the satellite moving fastest?",
                    question_type="multiple_choice",
                    options=[
                        "At aphelion (farthest point)",
                        "At perihelion (closest point)",
                        "Speed is constant",
                        "At the center of the ellipse"
                    ],
                    correct_answer="At perihelion (closest point)",
                    explanation="Conservation of angular momentum: L = mvr = constant. At closest approach (smallest r), velocity v must be largest.",
                    physics_concept="angular_momentum",
                    difficulty=2
                )
            ],
            "black_holes": [
                QuizQuestion(
                    id="bh_01",
                    question="What is the Schwarzschild radius for a black hole with 10 solar masses?",
                    question_type="numerical",
                    options=[],
                    correct_answer="30",
                    explanation="Rs = 2GM/c² ≈ 3 km per solar mass. For 10 solar masses: Rs ≈ 30 km.",
                    physics_concept="event_horizon",
                    difficulty=4
                ),
                QuizQuestion(
                    id="bh_02",
                    question="What happens to time near a black hole?",
                    question_type="multiple_choice",
                    options=[
                        "Time speeds up",
                        "Time slows down",
                        "Time flows backward",
                        "Time stops completely"
                    ],
                    correct_answer="Time slows down",
                    explanation="Gravitational time dilation: time runs slower in stronger gravitational fields.",
                    physics_concept="time_dilation",
                    difficulty=3
                )
            ]
        }
    
    def _init_physics_concepts(self):
        """Initialize mapping of physics concepts to explanations"""
        self.physics_concepts = {
            "inverse_square_law": {
                "title": "Inverse Square Law",
                "description": "Gravitational force decreases as the square of distance",
                "formula": "F = G·m₁·m₂/r²",
                "applications": ["planetary orbits", "tidal forces", "satellite motion"],
                "visualization_hints": ["field strength", "potential contours"]
            },
            "gravitational_field": {
                "title": "Gravitational Field",
                "description": "Region of space where gravitational force is experienced",
                "formula": "g = GM/r²",
                "applications": ["weight calculation", "field mapping", "superposition"],
                "visualization_hints": ["vector field", "field lines"]
            },
            "keplers_laws": {
                "title": "Kepler's Laws",
                "description": "Three laws describing planetary motion",
                "formula": "T² = (4π²/GM)·r³",
                "applications": ["orbital prediction", "planet discovery", "mission planning"],
                "visualization_hints": ["elliptical orbits", "period-radius relationship"]
            },
            "event_horizon": {
                "title": "Event Horizon",
                "description": "Boundary around black hole from which nothing can escape",
                "formula": "Rs = 2GM/c²",
                "applications": ["black hole physics", "information paradox", "hawking radiation"],
                "visualization_hints": ["spherical boundary", "light cones"]
            },
            "angular_momentum": {
                "title": "Angular Momentum Conservation",
                "description": "Rotational momentum is conserved in orbital motion",
                "formula": "L = r × mv = constant",
                "applications": ["orbital mechanics", "planetary rings", "galaxy formation"],
                "visualization_hints": ["orbital eccentricity", "velocity variation"]
            }
        }
    
    def get_learning_path(self, user_level: int = 1) -> List[Dict[str, Any]]:
        """
        Generate adaptive learning path based on user level.
        
        Args:
            user_level: Current user proficiency level (1-5)
            
        Returns:
            Ordered list of learning objectives
        """
        # Sort objectives by difficulty and prerequisites
        available_objectives = []
        completed_prerequisites = set()
        
        # Start with beginner objectives
        if user_level >= 1:
            completed_prerequisites.add("gravity_basics")
        if user_level >= 2:
            completed_prerequisites.add("orbital_mechanics")
        if user_level >= 3:
            completed_prerequisites.add("advanced_gravity")
        if user_level >= 4:
            completed_prerequisites.add("black_holes")
        
        learning_path = []
        remaining_objectives = dict(self.learning_objectives)
        
        while remaining_objectives:
            # Find objectives with satisfied prerequisites
            ready_objectives = []
            for obj_id, obj in remaining_objectives.items():
                if all(prereq in completed_prerequisites for prereq in obj.prerequisites):
                    ready_objectives.append((obj_id, obj))
            
            if not ready_objectives:
                break  # No more objectives can be unlocked
            
            # Sort by difficulty level
            ready_objectives.sort(key=lambda x: x[1].difficulty_level)
            
            # Add the next objective
            next_obj_id, next_obj = ready_objectives[0]
            learning_path.append({
                "id": next_obj_id,
                "title": next_obj.title,
                "description": next_obj.description,
                "difficulty": next_obj.difficulty_level,
                "concepts": next_obj.physics_concepts,
                "scenarios": next_obj.simulation_scenarios,
                "estimated_time": self._estimate_completion_time(next_obj)
            })
            
            # Mark as completed for prerequisite checking
            completed_prerequisites.add(next_obj_id)
            del remaining_objectives[next_obj_id]
        
        return learning_path
    
    def _estimate_completion_time(self, objective: LearningObjective) -> int:
        """Estimate completion time in minutes"""
        base_time = 15  # Base time per objective
        difficulty_multiplier = objective.difficulty_level
        concept_time = len(objective.physics_concepts) * 5
        simulation_time = len(objective.simulation_scenarios) * 10
        
        return base_time + (difficulty_multiplier * 10) + concept_time + simulation_time
    
    def update_user_progress(self, user_id: str, lesson_id: str, progress_data: Dict[str, Any]):
        """Update user progress in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Convert quiz scores to JSON string
        quiz_scores_json = json.dumps(progress_data.get('quiz_scores', []))
        
        cursor.execute('''
            INSERT OR REPLACE INTO user_progress 
            (user_id, lesson_id, completion_percentage, time_spent, last_accessed, 
             quiz_scores, simulation_interactions, mastery_level, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            user_id, lesson_id,
            progress_data.get('completion_percentage', 0),
            progress_data.get('time_spent', 0),
            datetime.now().isoformat(),
            quiz_scores_json,
            progress_data.get('simulation_interactions', 0),
            progress_data.get('mastery_level', 0),
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def get_user_progress(self, user_id: str) -> Dict[str, UserProgress]:
        """Get all progress data for a user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM user_progress WHERE user_id = ?
        ''', (user_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        progress_data = {}
        for row in rows:
            quiz_scores = json.loads(row[6]) if row[6] else []
            progress = UserProgress(
                user_id=row[1],
                lesson_id=row[2],
                completion_percentage=row[3],
                time_spent=row[4],
                last_accessed=row[5],
                quiz_scores=quiz_scores,
                simulation_interactions=row[7],
                mastery_level=row[8]
            )
            progress_data[row[2]] = progress
        
        return progress_data
    
    def generate_quiz(self, topic: str, difficulty_range: Tuple[int, int] = (1, 5), num_questions: int = 5) -> List[QuizQuestion]:
        """
        Generate adaptive quiz for a topic.
        
        Args:
            topic: Topic identifier
            difficulty_range: Range of difficulty levels to include
            num_questions: Number of questions to generate
            
        Returns:
            List of quiz questions
        """
        if topic not in self.quiz_questions:
            return []
        
        available_questions = [
            q for q in self.quiz_questions[topic]
            if difficulty_range[0] <= q.difficulty <= difficulty_range[1]
        ]
        
        # Randomize and select
        np.random.shuffle(available_questions)
        return available_questions[:num_questions]
    
    def record_quiz_attempt(self, user_id: str, quiz_id: str, question_id: str, 
                           user_answer: str, correct_answer: str, response_time: float):
        """Record a quiz attempt"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        is_correct = user_answer.lower().strip() == correct_answer.lower().strip()
        
        cursor.execute('''
            INSERT INTO quiz_attempts 
            (user_id, quiz_id, question_id, user_answer, correct_answer, is_correct, response_time)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (user_id, quiz_id, question_id, user_answer, correct_answer, is_correct, response_time))
        
        conn.commit()
        conn.close()
        
        return is_correct
    
    def get_user_insights(self, user_id: str) -> Dict[str, Any]:
        """Generate learning insights for a user"""
        progress_data = self.get_user_progress(user_id)
        
        if not progress_data:
            return {
                "message": "No learning data available yet. Start with some lessons!",
                "recommendations": ["Begin with 'Introduction to Gravity'"]
            }
        
        # Calculate overall statistics
        total_lessons = len(progress_data)
        completed_lessons = sum(1 for p in progress_data.values() if p.completion_percentage >= 100)
        avg_completion = np.mean([p.completion_percentage for p in progress_data.values()])
        total_time = sum(p.time_spent for p in progress_data.values())
        avg_mastery = np.mean([p.mastery_level for p in progress_data.values()])
        
        # Identify strengths and weaknesses
        quiz_performance = self._analyze_quiz_performance(user_id)
        concept_mastery = self._analyze_concept_mastery(user_id, progress_data)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(user_id, progress_data, concept_mastery)
        
        insights = {
            "overall_progress": {
                "completed_lessons": completed_lessons,
                "total_lessons": total_lessons,
                "completion_rate": completed_lessons / total_lessons if total_lessons > 0 else 0,
                "avg_completion": avg_completion,
                "total_time_hours": total_time / 60,
                "avg_mastery": avg_mastery
            },
            "quiz_performance": quiz_performance,
            "concept_mastery": concept_mastery,
            "recommendations": recommendations,
            "learning_streaks": self._calculate_learning_streaks(user_id),
            "next_objectives": self._get_next_objectives(user_id, progress_data)
        }
        
        return insights
    
    def _analyze_quiz_performance(self, user_id: str) -> Dict[str, Any]:
        """Analyze quiz performance patterns"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT quiz_id, is_correct, response_time, timestamp 
            FROM quiz_attempts 
            WHERE user_id = ?
            ORDER BY timestamp DESC
        ''', (user_id,))
        
        attempts = cursor.fetchall()
        conn.close()
        
        if not attempts:
            return {"message": "No quiz attempts yet"}
        
        # Calculate accuracy by topic
        topic_accuracy = {}
        response_times = []
        
        for quiz_id, is_correct, response_time, timestamp in attempts:
            topic = quiz_id.split('_')[0] if '_' in quiz_id else quiz_id
            
            if topic not in topic_accuracy:
                topic_accuracy[topic] = []
            
            topic_accuracy[topic].append(is_correct)
            if response_time:
                response_times.append(response_time)
        
        # Calculate statistics
        overall_accuracy = np.mean([attempt[1] for attempt in attempts])
        avg_response_time = np.mean(response_times) if response_times else 0
        
        topic_stats = {}
        for topic, scores in topic_accuracy.items():
            topic_stats[topic] = {
                "accuracy": np.mean(scores),
                "attempts": len(scores),
                "trend": "improving" if len(scores) > 3 and np.mean(scores[-3:]) > np.mean(scores[:-3]) else "stable"
            }
        
        return {
            "overall_accuracy": overall_accuracy,
            "avg_response_time": avg_response_time,
            "topic_performance": topic_stats,
            "total_attempts": len(attempts)
        }
    
    def _analyze_concept_mastery(self, user_id: str, progress_data: Dict[str, UserProgress]) -> Dict[str, float]:
        """Analyze mastery of different physics concepts"""
        concept_scores = {}
        
        for lesson_id, progress in progress_data.items():
            if lesson_id in self.learning_objectives:
                objective = self.learning_objectives[lesson_id]
                for concept in objective.physics_concepts:
                    if concept not in concept_scores:
                        concept_scores[concept] = []
                    
                    # Weight by completion and quiz performance
                    concept_score = (
                        progress.completion_percentage / 100 * 0.4 +
                        progress.mastery_level * 0.6
                    )
                    concept_scores[concept].append(concept_score)
        
        # Average scores for each concept
        concept_mastery = {}
        for concept, scores in concept_scores.items():
            concept_mastery[concept] = np.mean(scores)
        
        return concept_mastery
    
    def _generate_recommendations(self, user_id: str, progress_data: Dict[str, UserProgress], 
                                concept_mastery: Dict[str, float]) -> List[str]:
        """Generate personalized learning recommendations"""
        recommendations = []
        
        # Find weak concepts
        weak_concepts = [
            concept for concept, mastery in concept_mastery.items()
            if mastery < 0.7
        ]
        
        if weak_concepts:
            recommendations.append(
                f"Focus on improving understanding of: {', '.join(weak_concepts[:3])}"
            )
        
        # Find incomplete lessons
        incomplete_lessons = [
            lesson_id for lesson_id, progress in progress_data.items()
            if progress.completion_percentage < 100
        ]
        
        if incomplete_lessons:
            recommendations.append(
                f"Complete these lessons: {', '.join(incomplete_lessons[:2])}"
            )
        
        # Time-based recommendations
        recent_activity = any(
            datetime.fromisoformat(progress.last_accessed) > datetime.now() - timedelta(days=7)
            for progress in progress_data.values()
        )
        
        if not recent_activity:
            recommendations.append("You haven't practiced in a while. Try a quick review!")
        
        # Difficulty progression
        avg_mastery = np.mean(list(concept_mastery.values())) if concept_mastery else 0
        if avg_mastery > 0.8:
            recommendations.append("Great progress! Ready for more advanced topics.")
        elif avg_mastery < 0.5:
            recommendations.append("Take your time with fundamentals before advancing.")
        
        return recommendations if recommendations else ["Keep up the great work!"]
    
    def _calculate_learning_streaks(self, user_id: str) -> Dict[str, int]:
        """Calculate learning streaks and patterns"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT DISTINCT DATE(timestamp) as date
            FROM learning_analytics 
            WHERE user_id = ?
            ORDER BY date DESC
        ''', (user_id,))
        
        dates = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        if not dates:
            return {"current_streak": 0, "longest_streak": 0}
        
        # Calculate current streak
        current_streak = 0
        current_date = datetime.now().date()
        
        for date_str in dates:
            date = datetime.fromisoformat(date_str).date()
            if (current_date - date).days == current_streak:
                current_streak += 1
                current_date = date
            else:
                break
        
        # Calculate longest streak
        longest_streak = 1
        current_run = 1
        
        for i in range(1, len(dates)):
            prev_date = datetime.fromisoformat(dates[i-1]).date()
            curr_date = datetime.fromisoformat(dates[i]).date()
            
            if (prev_date - curr_date).days == 1:
                current_run += 1
                longest_streak = max(longest_streak, current_run)
            else:
                current_run = 1
        
        return {
            "current_streak": current_streak,
            "longest_streak": longest_streak,
            "total_active_days": len(dates)
        }
    
    def _get_next_objectives(self, user_id: str, progress_data: Dict[str, UserProgress]) -> List[Dict[str, Any]]:
        """Get next recommended learning objectives"""
        completed_lessons = set(
            lesson_id for lesson_id, progress in progress_data.items()
            if progress.completion_percentage >= 80  # Consider 80% as completed
        )
        
        next_objectives = []
        for obj_id, objective in self.learning_objectives.items():
            if obj_id not in completed_lessons:
                # Check if prerequisites are met
                prerequisites_met = all(
                    prereq in completed_lessons for prereq in objective.prerequisites
                )
                
                if prerequisites_met:
                    next_objectives.append({
                        "id": obj_id,
                        "title": objective.title,
                        "difficulty": objective.difficulty_level,
                        "estimated_time": self._estimate_completion_time(objective)
                    })
        
        # Sort by difficulty
        next_objectives.sort(key=lambda x: x["difficulty"])
        return next_objectives[:3]  # Return top 3 recommendations
    
    def log_learning_event(self, user_id: str, event_type: str, event_data: Dict[str, Any], session_id: str):
        """Log learning analytics events"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO learning_analytics (user_id, event_type, event_data, session_id)
            VALUES (?, ?, ?, ?)
        ''', (user_id, event_type, json.dumps(event_data), session_id))
        
        conn.commit()
        conn.close()
    
    def get_concept_explanation(self, concept_id: str) -> Dict[str, Any]:
        """Get detailed explanation for a physics concept"""
        if concept_id in self.physics_concepts:
            return self.physics_concepts[concept_id]
        else:
            return {
                "title": concept_id.replace("_", " ").title(),
                "description": "Physics concept explanation not available",
                "formula": "",
                "applications": [],
                "visualization_hints": []
            }
    
    def generate_adaptive_content(self, user_id: str, topic: str) -> Dict[str, Any]:
        """Generate adaptive content based on user's current level"""
        user_insights = self.get_user_insights(user_id)
        concept_mastery = user_insights.get("concept_mastery", {})
        
        # Determine appropriate difficulty level
        avg_mastery = np.mean(list(concept_mastery.values())) if concept_mastery else 0.5
        
        if avg_mastery < 0.3:
            difficulty_level = 1
            content_focus = "basic_concepts"
        elif avg_mastery < 0.6:
            difficulty_level = 2
            content_focus = "application"
        elif avg_mastery < 0.8:
            difficulty_level = 3
            content_focus = "analysis"
        else:
            difficulty_level = 4
            content_focus = "synthesis"
        
        # Generate adaptive quiz
        quiz = self.generate_quiz(topic, (max(1, difficulty_level-1), difficulty_level+1), 3)
        
        # Get relevant concepts
        relevant_concepts = []
        if topic in self.learning_objectives:
            for concept in self.learning_objectives[topic].physics_concepts:
                relevant_concepts.append(self.get_concept_explanation(concept))
        
        return {
            "difficulty_level": difficulty_level,
            "content_focus": content_focus,
            "quiz_questions": quiz,
            "relevant_concepts": relevant_concepts,
            "recommended_simulations": self.learning_objectives.get(topic, LearningObjective("", "", "", [], 1, [], [])).simulation_scenarios
        }
    
    def get_beginner_friendly_content(self, topic: str, difficulty_level: int = 1) -> Dict[str, Any]:
        """Get beginner-friendly educational content with progressive difficulty"""
        
        beginner_content = {
            "gravity_basics": {
                1: {
                    "title": "What is Gravity? 🍎",
                    "simple_explanation": """
                    Imagine gravity as an invisible hand that pulls everything down to Earth. 
                    When you drop an apple, gravity pulls it toward the ground. The bigger 
                    something is, the stronger its gravity!
                    """,
                    "analogy": "Like a magnet, but it works on everything, not just metal!",
                    "interactive_demo": "apple_drop_simple",
                    "fun_facts": [
                        "Gravity keeps your feet on the ground",
                        "Without gravity, you would float like an astronaut",
                        "The Moon's gravity is 6 times weaker than Earth's"
                    ],
                    "next_level": "Learn about gravity's strength and the Universal Law"
                },
                2: {
                    "title": "Newton's Universal Law of Gravitation ⚖️",
                    "simple_explanation": """
                    Isaac Newton discovered that EVERYTHING with mass attracts everything else.
                    This includes you, your desk, the Earth, and even distant stars! The force
                    depends on two things: how massive the objects are, and how far apart they are.
                    """,
                    "analogy": "Like invisible rubber bands connecting all objects in the universe",
                    "interactive_demo": "force_calculator",
                    "key_points": [
                        "Bigger masses = stronger gravity",
                        "Closer distance = stronger gravity", 
                        "Gravity gets weaker very quickly with distance"
                    ],
                    "formula_intro": "F = G × (mass1 × mass2) / distance²",
                    "next_level": "Explore how gravity creates orbits"
                }
            },
            
            "black_holes": {
                1: {
                    "title": "Black Holes - The Universe's Vacuum Cleaners? 🕳️",
                    "simple_explanation": """
                    Black holes are NOT vacuum cleaners that suck everything up! They're actually
                    regions where gravity is so strong that not even light can escape. Think of
                    them as really, really heavy objects that bend space around them.
                    """,
                    "common_misconceptions": {
                        "❌ Black holes suck everything up": "✅ Only affects nearby objects",
                        "❌ Black holes are empty space": "✅ Contain enormous amounts of matter",
                        "❌ Falling into one would be instant death": "✅ You'd stretch out like spaghetti first!"
                    },
                    "analogy": "Like a bowling ball on a trampoline - it creates a deep dip that marbles roll into",
                    "interactive_demo": "black_hole_simulator_simple",
                    "fun_facts": [
                        "There's a supermassive black hole at the center of our galaxy",
                        "Time actually slows down near black holes",
                        "The first black hole photo was taken in 2019"
                    ]
                },
                2: {
                    "title": "Event Horizons and Spacetime 🌌",
                    "simple_explanation": """
                    The event horizon is like a one-way door around a black hole. Once you cross it,
                    you can never come back out. It's not a physical surface - just a boundary in space
                    where the escape velocity becomes faster than light.
                    """,
                    "visual_explanation": "Imagine trying to throw a ball so hard it never falls back down",
                    "interactive_demo": "event_horizon_explorer",
                    "time_dilation_simple": "Near black holes, time moves differently - like slow motion!"
                }
            },
            
            "orbital_mechanics": {
                1: {
                    "title": "Why Don't Satellites Fall Down? 🛰️",
                    "simple_explanation": """
                    Satellites are actually falling all the time! But they're moving so fast sideways
                    that they keep missing the Earth. It's like throwing a ball really, really hard -
                    eventually it would go all the way around the world!
                    """,
                    "analogy": "Like a ball on a string - centrifugal force keeps it from falling in",
                    "interactive_demo": "orbital_motion_simple",
                    "key_insight": "Orbiting = falling forward fast enough to miss the ground",
                    "real_examples": [
                        "International Space Station circles Earth every 90 minutes",
                        "Moon takes 27 days to orbit Earth",
                        "GPS satellites orbit twice a day"
                    ]
                }
            },
            
            "relativity": {
                1: {
                    "title": "Einstein's Wild Ideas About Time and Space ⏰",
                    "simple_explanation": """
                    Einstein discovered something amazing: time isn't the same everywhere! If you
                    travel really fast or go near something really heavy, time actually slows down
                    for you compared to everyone else.
                    """,
                    "everyday_examples": [
                        "GPS satellites have to correct their clocks because time runs differently in space",
                        "If you traveled to a star and back at light speed, everyone on Earth would age more than you",
                        "Gravity actually makes time run slower"
                    ],
                    "interactive_demo": "time_dilation_simulator",
                    "mind_bending_fact": "Light always travels at the same speed, no matter how fast you're moving!"
                }
            }
        }
        
        return beginner_content.get(topic, {}).get(difficulty_level, {})
    
    def create_progressive_learning_path(self, user_level: str = "beginner") -> List[Dict[str, Any]]:
        """Create a progressive learning path based on user level"""
        
        learning_paths = {
            "beginner": [
                {
                    "module": "gravity_basics",
                    "level": 1,
                    "estimated_time": "15 min",
                    "prerequisites": None,
                    "description": "Start with the basics - what is gravity?",
                    "activities": ["Watch animations", "Try interactive demos", "Take simple quiz"]
                },
                {
                    "module": "gravity_basics", 
                    "level": 2,
                    "estimated_time": "20 min",
                    "prerequisites": ["gravity_basics_1"],
                    "description": "Learn Newton's law of universal gravitation",
                    "activities": ["Force calculator", "Compare planetary gravity", "Mini-quiz"]
                },
                {
                    "module": "orbital_mechanics",
                    "level": 1, 
                    "estimated_time": "25 min",
                    "prerequisites": ["gravity_basics_2"],
                    "description": "Understand how satellites stay in orbit",
                    "activities": ["Orbit simulator", "Design a satellite mission", "Orbital quiz"]
                },
                {
                    "module": "black_holes",
                    "level": 1,
                    "estimated_time": "30 min", 
                    "prerequisites": ["orbital_mechanics_1"],
                    "description": "Explore the mysteries of black holes",
                    "activities": ["Black hole myths vs facts", "Event horizon explorer", "Time dilation demo"]
                },
                {
                    "module": "relativity",
                    "level": 1,
                    "estimated_time": "35 min",
                    "prerequisites": ["black_holes_1"],
                    "description": "Einstein's revolutionary ideas about space and time",
                    "activities": ["GPS relativity game", "Twin paradox simulator", "Relativity quiz"]
                }
            ],
            
            "intermediate": [
                {
                    "module": "advanced_orbits",
                    "level": 1,
                    "estimated_time": "45 min",
                    "prerequisites": ["beginner_path_complete"],
                    "description": "Lagrange points and multi-body systems",
                    "activities": ["Lagrange point explorer", "Mission planning", "Stability analysis"]
                },
                {
                    "module": "general_relativity",
                    "level": 1, 
                    "estimated_time": "50 min",
                    "prerequisites": ["advanced_orbits_1"],
                    "description": "Curved spacetime and gravitational effects",
                    "activities": ["Spacetime visualizer", "Gravitational lensing", "Frame dragging demo"]
                }
            ],
            
            "advanced": [
                {
                    "module": "gravitational_waves",
                    "level": 1,
                    "estimated_time": "60 min", 
                    "prerequisites": ["intermediate_path_complete"],
                    "description": "Ripples in spacetime from cosmic events",
                    "activities": ["LIGO data analysis", "Binary merger simulation", "Wave detection game"]
                },
                {
                    "module": "exotic_physics",
                    "level": 1,
                    "estimated_time": "75 min",
                    "prerequisites": ["gravitational_waves_1"], 
                    "description": "Wormholes, dark matter, and beyond",
                    "activities": ["Wormhole navigator", "Dark matter simulation", "Exotic matter calculator"]
                }
            ]
        }
        
        return learning_paths.get(user_level, learning_paths["beginner"])
    
    def generate_adaptive_hints(self, topic: str, user_performance: Dict[str, float]) -> List[str]:
        """Generate adaptive hints based on user performance"""
        
        performance_score = user_performance.get(topic, 0.0)
        
        hint_database = {
            "gravity_basics": {
                "struggling": [
                    "💡 Think of gravity like invisible glue holding everything together",
                    "🍎 Try the apple drop experiment - watch how objects fall",
                    "🌍 Remember: bigger objects have stronger gravity",
                    "📐 Distance matters a lot - gravity gets weaker quickly as you move away"
                ],
                "progressing": [
                    "⚖️ Try calculating gravity between different objects",
                    "🎯 Focus on how mass and distance affect gravitational force",
                    "🔢 Practice with the formula: F = G × m₁ × m₂ / r²",
                    "🌙 Compare Earth's gravity with the Moon's gravity"
                ],
                "mastering": [
                    "🚀 Ready for orbital mechanics? See how gravity creates orbits",
                    "🌟 Explore how gravity works between stars and planets",
                    "⚡ Challenge yourself with escape velocity calculations"
                ]
            },
            
            "black_holes": {
                "struggling": [
                    "🕳️ Remember: black holes don't suck things up like vacuum cleaners",
                    "🥣 Imagine spacetime like a stretchy fabric that gets bent by heavy objects",
                    "⚫ The event horizon is just a boundary, not a physical surface",
                    "🍝 'Spaghettification' happens because gravity is stronger at your feet than your head"
                ],
                "progressing": [
                    "🧮 Try calculating the Schwarzschild radius for different masses",
                    "⏰ Explore how time dilation works near black holes",
                    "🌌 Compare stellar black holes with supermassive ones",
                    "🔭 Learn about real black hole observations"
                ],
                "mastering": [
                    "🌀 Ready for rotating black holes and frame dragging effects?",
                    "📡 Explore Hawking radiation and black hole thermodynamics",
                    "🌉 Connect to wormhole physics and exotic spacetime geometries"
                ]
            }
        }
        
        if performance_score < 0.3:
            difficulty = "struggling"
        elif performance_score < 0.7:
            difficulty = "progressing"  
        else:
            difficulty = "mastering"
            
        return hint_database.get(topic, {}).get(difficulty, ["Keep practicing!"])
    
    def create_interactive_glossary(self) -> Dict[str, Dict[str, str]]:
        """Create an interactive glossary with beginner-friendly definitions"""
        
        return {
            "acceleration": {
                "simple": "How quickly something speeds up or slows down",
                "example": "When you press the gas pedal, your car accelerates",
                "unit": "meters per second squared (m/s²)"
            },
            "black_hole": {
                "simple": "A region in space where gravity is so strong that nothing can escape",
                "example": "Like a cosmic drain, but it doesn't suck - things have to get very close",
                "misconception": "They don't vacuum up everything around them"
            },
            "escape_velocity": {
                "simple": "How fast you need to throw something to escape gravity completely",
                "example": "On Earth, you'd need to throw a ball at 11 km/s (25,000 mph)!",
                "fun_fact": "On the Moon, you only need 2.4 km/s because gravity is weaker"
            },
            "event_horizon": {
                "simple": "The invisible boundary around a black hole - the point of no return",
                "example": "Like crossing a one-way bridge - you can't come back",
                "size": "Depends on the black hole's mass - bigger mass = bigger event horizon"
            },
            "gravity": {
                "simple": "The force that pulls objects with mass toward each other", 
                "example": "What keeps your feet on the ground and makes things fall down",
                "universal": "Everything with mass has gravity - even you!"
            },
            "lagrange_point": {
                "simple": "Special parking spots in space where gravitational forces balance perfectly",
                "example": "Like balancing on a see-saw between Earth and Sun",
                "applications": "Perfect places for space telescopes and satellites"
            },
            "orbital_velocity": {
                "simple": "How fast something needs to move to stay in orbit around a planet",
                "example": "The ISS travels at 27,600 km/h to stay in orbit around Earth",
                "balance": "Fast enough to not fall down, but not so fast it flies away"
            },
            "time_dilation": {
                "simple": "Time running at different speeds in different places",
                "example": "Clocks run slightly slower on GPS satellites",
                "cause": "Happens when you move very fast or are near strong gravity"
            },
            "wormhole": {
                "simple": "A theoretical tunnel through space that could connect distant places",
                "example": "Like a shortcut through a folded piece of paper",
                "reality": "Pure theory - we've never found or created one"
            }
        }
