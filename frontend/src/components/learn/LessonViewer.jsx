import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  Box,
  Container,
  Typography,
  Paper,
  Stepper,
  Step,
  StepLabel,
  Button,
  Divider,
  Card,
  CardContent,
  LinearProgress,
  Chip
} from '@mui/material';
import {
  NavigateNext,
  NavigateBefore,
  CheckCircle,
  PlayCircle
} from '@mui/icons-material';
import { motion } from 'framer-motion';
import ReactMarkdown from 'react-markdown';
import InteractiveDemo from './InteractiveDemo';
import { useGameStore } from '../../store/gameStore';

// Lesson content would typically come from a CMS or markdown files
const lessonContent = {
  'newton-basics': {
    title: "Newton's Law of Universal Gravitation",
    sections: [
      {
        title: "The Apple That Changed Physics",
        content: `
# The Apple That Changed Physics

In 1666, a young Isaac Newton was sitting in his garden when he saw an apple fall from a tree. This simple observation led to one of the most profound discoveries in physics.

Newton realized that the same force that pulled the apple to Earth also kept the Moon in orbit. This was revolutionary - it unified earthly and celestial mechanics.

## Key Insight
The force of gravity is **universal** - it acts between all objects with mass, everywhere in the universe.
        `,
        demo: 'apple-drop',
        quiz: [
          {
            question: "What did Newton realize about the apple and the Moon?",
            options: [
              "They are made of the same material",
              "They are affected by the same force",
              "They have the same mass",
              "They fall at the same speed"
            ],
            correct: 1
          }
        ]
      },
      {
        title: "Understanding Force and Mass",
        content: `
# Understanding Force and Mass

Newton's law states that every particle attracts every other particle with a force that is:
- **Proportional** to the product of their masses
- **Inversely proportional** to the square of the distance between them

## The Formula
\`F = G × (m₁ × m₂) / r²\`

Where:
- F = gravitational force (N)
- G = gravitational constant (6.674 × 10⁻¹¹ N⋅m²/kg²)
- m₁, m₂ = masses of the objects (kg)
- r = distance between centers (m)
        `,
        demo: 'force-calculator',
        quiz: [
          {
            question: "If you double the distance between two objects, the gravitational force becomes:",
            options: [
              "Half as strong",
              "Twice as strong",
              "One quarter as strong",
              "Four times as strong"
            ],
            correct: 2
          }
        ]
      }
    ]
  }
  // Add more lessons...
};

const LessonViewer = () => {
  const { lessonId } = useParams();
  const navigate = useNavigate();
  const [currentSection, setCurrentSection] = useState(0);
  const [sectionProgress, setSectionProgress] = useState({});
  const [showDemo, setShowDemo] = useState(false);
  const { addScore, addAchievement } = useGameStore();
  
  const lesson = lessonContent[lessonId];
  const section = lesson?.sections[currentSection];
  
  useEffect(() => {
    // Track lesson progress
    if (section) {
      setSectionProgress(prev => ({
        ...prev,
        [currentSection]: true
      }));
    }
  }, [currentSection, section]);
  
  const handleNext = () => {
    if (currentSection < lesson.sections.length - 1) {
      setCurrentSection(currentSection + 1);
      window.scrollTo(0, 0);
    } else {
      // Lesson completed
      addScore(50);
      addAchievement(`Completed: ${lesson.title}`);
      navigate('/learn');
    }
  };
  
  const handlePrevious = () => {
    if (currentSection > 0) {
      setCurrentSection(currentSection - 1);
      window.scrollTo(0, 0);
    }
  };
  
  const handleQuizAnswer = (questionIndex, answerIndex) => {
    const isCorrect = section.quiz[questionIndex].correct === answerIndex;
    if (isCorrect) {
      addScore(10);
    }
    // Update UI to show correct/incorrect
  };
  
  if (!lesson) {
    return (
      <Container>
        <Typography>Lesson not found</Typography>
      </Container>
    );
  }
  
  const progress = (Object.keys(sectionProgress).length / lesson.sections.length) * 100;
  
  return (
    <Container maxWidth="md" sx={{ py: 4 }}>
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        {/* Header */}
        <Paper sx={{ p: 3, mb: 3 }}>
          <Typography variant="h4" gutterBottom>
            {lesson.title}
          </Typography>
          <LinearProgress 
            variant="determinate" 
            value={progress} 
            sx={{ mb: 2 }}
          />
          <Stepper activeStep={currentSection} alternativeLabel>
            {lesson.sections.map((section, index) => (
              <Step key={index} completed={sectionProgress[index]}>
                <StepLabel>{section.title}</StepLabel>
              </Step>
            ))}
          </Stepper>
        </Paper>
        
        {/* Content */}
        <Paper sx={{ p: 4, mb: 3 }}>
          <Box className="markdown-content">
            <ReactMarkdown>{section.content}</ReactMarkdown>
          </Box>
          
          {/* Interactive Demo */}
          {section.demo && (
            <Box sx={{ my: 4 }}>
              <Button
                variant="contained"
                startIcon={<PlayCircle />}
                onClick={() => setShowDemo(!showDemo)}
                fullWidth
              >
                {showDemo ? 'Hide' : 'Show'} Interactive Demo
              </Button>
              
              {showDemo && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  transition={{ duration: 0.3 }}
                >
                  <Card sx={{ mt: 2 }}>
                    <CardContent>
                      <InteractiveDemo type={section.demo} />
                    </CardContent>
                  </Card>
                </motion.div>
              )}
            </Box>
          )}
          
          {/* Quiz */}
          {section.quiz && (
            <Box sx={{ mt: 4 }}>
              <Typography variant="h6" gutterBottom>
                Quick Check
              </Typography>
              {section.quiz.map((q, qIndex) => (
                <Card key={qIndex} sx={{ mb: 2 }}>
                  <CardContent>
                    <Typography variant="body1" gutterBottom>
                      {q.question}
                    </Typography>
                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                      {q.options.map((option, oIndex) => (
                        <Button
                          key={oIndex}
                          variant="outlined"
                          onClick={() => handleQuizAnswer(qIndex, oIndex)}
                          sx={{ justifyContent: 'flex-start' }}
                        >
                          {option}
                        </Button>
                      ))}
                    </Box>
                  </CardContent>
                </Card>
              ))}
            </Box>
          )}
        </Paper>
        
        {/* Navigation */}
        <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
          <Button
            variant="outlined"
            startIcon={<NavigateBefore />}
            onClick={handlePrevious}
            disabled={currentSection === 0}
          >
            Previous
          </Button>
          
          <Button
            variant="contained"
            endIcon={currentSection === lesson.sections.length - 1 ? <CheckCircle /> : <NavigateNext />}
            onClick={handleNext}
          >
            {currentSection === lesson.sections.length - 1 ? 'Complete Lesson' : 'Next Section'}
          </Button>
        </Box>
      </motion.div>
    </Container>
  );
};

export default LessonViewer;