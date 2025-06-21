import React, { useState } from 'react';
import {
  Box,
  Container,
  Typography,
  Paper,
  Button,
  RadioGroup,
  FormControlLabel,
  Radio,
  LinearProgress,
  Card,
  CardContent,
  Chip,
  Alert
} from '@mui/material';
import { motion, AnimatePresence } from 'framer-motion';
import { CheckCircle, Cancel, NavigateNext } from '@mui/icons-material';
import { useGameStore } from '../../store/gameStore';

const quizQuestions = [
  {
    id: 1,
    category: 'Newton',
    difficulty: 'easy',
    question: 'According to Newton\'s law of universal gravitation, the force between two objects is:',
    options: [
      'Proportional to the distance between them',
      'Inversely proportional to the distance between them',
      'Inversely proportional to the square of the distance between them',
      'Independent of the distance between them'
    ],
    correct: 2,
    explanation: 'The gravitational force follows an inverse square law: F ∝ 1/r²'
  },
  {
    id: 2,
    category: 'Orbits',
    difficulty: 'medium',
    question: 'What happens to the orbital period of a satellite if its orbital radius is doubled?',
    options: [
      'It doubles',
      'It increases by √2',
      'It increases by 2√2',
      'It quadruples'
    ],
    correct: 2,
    explanation: 'According to Kepler\'s third law, T² ∝ r³, so if r doubles, T increases by 2^(3/2) = 2√2'
  },
  {
    id: 3,
    category: 'Energy',
    difficulty: 'medium',
    question: 'At escape velocity, the total mechanical energy of an object is:',
    options: [
      'Positive',
      'Negative',
      'Zero',
      'Infinite'
    ],
    correct: 2,
    explanation: 'At escape velocity, kinetic energy exactly cancels gravitational potential energy, giving zero total energy'
  },
  {
    id: 4,
    category: 'Relativity',
    difficulty: 'hard',
    question: 'Near a black hole\'s event horizon, time dilation causes:',
    options: [
      'Time to speed up for the falling observer',
      'Time to slow down for the falling observer',
      'Time to slow down as seen by a distant observer',
      'No effect on time'
    ],
    correct: 2,
    explanation: 'From a distant observer\'s perspective, time slows down near the event horizon due to gravitational time dilation'
  },
  {
    id: 5,
    category: 'Tides',
    difficulty: 'easy',
    question: 'Ocean tides are primarily caused by:',
    options: [
      'The Sun\'s gravity alone',
      'The Moon\'s gravity alone',
      'The difference in gravitational force across Earth',
      'Earth\'s rotation alone'
    ],
    correct: 2,
    explanation: 'Tides are caused by the gradient (difference) in gravitational force across Earth\'s diameter'
  }
];

const QuizComponent = () => {
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [selectedAnswer, setSelectedAnswer] = useState(null);
  const [showResult, setShowResult] = useState(false);
  const [score, setScore] = useState(0);
  const [answeredQuestions, setAnsweredQuestions] = useState([]);
  const { addScore, addAchievement } = useGameStore();
  
  const question = quizQuestions[currentQuestion];
  const progress = ((currentQuestion + 1) / quizQuestions.length) * 100;
  
  const handleAnswer = () => {
    const isCorrect = selectedAnswer === question.correct;
    
    if (isCorrect) {
      setScore(score + 1);
      addScore(20);
    }
    
    setAnsweredQuestions([...answeredQuestions, {
      questionId: question.id,
      correct: isCorrect,
      selectedAnswer
    }]);
    
    setShowResult(true);
  };
  
  const handleNext = () => {
    if (currentQuestion < quizQuestions.length - 1) {
      setCurrentQuestion(currentQuestion + 1);
      setSelectedAnswer(null);
      setShowResult(false);
    } else {
      // Quiz completed
      const percentage = (score / quizQuestions.length) * 100;
      if (percentage >= 80) {
        addAchievement('Quiz Master');
      }
      // Show final results
    }
  };
  
  const getDifficultyColor = (difficulty) => {
    switch (difficulty) {
      case 'easy': return 'success';
      case 'medium': return 'warning';
      case 'hard': return 'error';
      default: return 'default';
    }
  };
  
  return (
    <Container maxWidth="md" sx={{ py: 4 }}>
      <Paper sx={{ p: 4 }}>
        <Typography variant="h4" gutterBottom align="center">
          Gravity Physics Quiz
        </Typography>
        
        <Box sx={{ mb: 4 }}>
          <LinearProgress variant="determinate" value={progress} sx={{ mb: 2 }} />
          <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
            <Typography variant="body2">
              Question {currentQuestion + 1} of {quizQuestions.length}
            </Typography>
            <Typography variant="body2">
              Score: {score}/{currentQuestion + (showResult ? 1 : 0)}
            </Typography>
          </Box>
        </Box>
        
        <AnimatePresence mode="wait">
          <motion.div
            key={currentQuestion}
            initial={{ opacity: 0, x: 50 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -50 }}
            transition={{ duration: 0.3 }}
          >
            <Card sx={{ mb: 4 }}>
              <CardContent>
                <Box sx={{ display: 'flex', gap: 1, mb: 2 }}>
                  <Chip 
                    label={question.category} 
                    size="small" 
                    variant="outlined"
                  />
                  <Chip 
                    label={question.difficulty} 
                    size="small" 
                    color={getDifficultyColor(question.difficulty)}
                  />
                </Box>
                
                <Typography variant="h6" gutterBottom>
                  {question.question}
                </Typography>
                
                <RadioGroup
                  value={selectedAnswer}
                  onChange={(e) => setSelectedAnswer(parseInt(e.target.value))}
                >
                  {question.options.map((option, index) => (
                    <FormControlLabel
                      key={index}
                      value={index}
                      control={<Radio />}
                      label={option}
                      disabled={showResult}
                      sx={{
                        backgroundColor: showResult
                          ? index === question.correct
                            ? 'success.light'
                            : index === selectedAnswer
                            ? 'error.light'
                            : 'transparent'
                          : 'transparent',
                        borderRadius: 1,
                        mb: 1,
                        pl: 1,
                        transition: 'background-color 0.3s'
                      }}
                    />
                  ))}
                </RadioGroup>
                
                {showResult && (
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3 }}
                  >
                    <Alert 
                      severity={selectedAnswer === question.correct ? 'success' : 'error'}
                      sx={{ mt: 2 }}
                      icon={selectedAnswer === question.correct ? <CheckCircle /> : <Cancel />}
                    >
                      <Typography variant="body2">
                        {selectedAnswer === question.correct ? 'Correct!' : 'Incorrect'}
                      </Typography>
                      <Typography variant="body2" sx={{ mt: 1 }}>
                        {question.explanation}
                      </Typography>
                    </Alert>
                  </motion.div>
                )}
              </CardContent>
            </Card>
            
            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
              <Button
                variant="contained"
                onClick={handleAnswer}
                disabled={selectedAnswer === null || showResult}
                sx={{ minWidth: 150 }}
              >
                Submit Answer
              </Button>
              
              <Button
                variant="outlined"
                endIcon={<NavigateNext />}
                onClick={handleNext}
                disabled={!showResult}
                sx={{ minWidth: 150 }}
              >
                {currentQuestion === quizQuestions.length - 1 ? 'Finish' : 'Next'}
              </Button>
            </Box>
          </motion.div>
        </AnimatePresence>
      </Paper>
    </Container>
  );
};

export default QuizComponent;