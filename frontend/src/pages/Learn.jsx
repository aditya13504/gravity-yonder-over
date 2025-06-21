import React, { useState } from 'react';
import { Routes, Route, Link, useNavigate } from 'react-router-dom';
import {
  Box,
  Container,
  Grid,
  Card,
  CardContent,
  Typography,
  List,
  ListItem,
  ListItemButton,
  ListItemText,
  Divider,
  Chip,
  Paper,
  Tabs,
  Tab,
  Button
} from '@mui/material';
import { motion } from 'framer-motion';
import {
  School,
  Science,
  Calculate,
  Quiz,
  VideoLibrary,
  ArrowForward
} from '@mui/icons-material';
import LessonViewer from '../components/learn/LessonViewer';
import InteractiveDemo from '../components/learn/InteractiveDemo';
import QuizComponent from '../components/learn/QuizComponent';

const lessons = [
  {
    id: 'newton-basics',
    title: "Newton's Law of Universal Gravitation",
    category: 'Classical Mechanics',
    difficulty: 'Beginner',
    topics: [
      'The Apple That Changed Physics',
      'Understanding Force and Mass',
      'The Inverse Square Law',
      'Calculating Gravitational Force'
    ],
    estimatedTime: '30 min'
  },
  {
    id: 'orbital-mechanics',
    title: 'Orbital Mechanics',
    category: 'Classical Mechanics',
    difficulty: 'Intermediate',
    topics: [
      "Kepler's Laws of Planetary Motion",
      'Circular vs Elliptical Orbits',
      'Orbital Velocity and Period',
      'Geostationary Orbits'
    ],
    estimatedTime: '45 min'
  },
  {
    id: 'escape-velocity',
    title: 'Escape Velocity and Energy',
    category: 'Classical Mechanics',
    difficulty: 'Intermediate',
    topics: [
      'Gravitational Potential Energy',
      'Kinetic Energy in Orbit',
      'Calculating Escape Velocity',
      'Energy Conservation'
    ],
    estimatedTime: '40 min'
  },
  {
    id: 'relativity-intro',
    title: 'Introduction to General Relativity',
    category: 'Modern Physics',
    difficulty: 'Advanced',
    topics: [
      'Spacetime Curvature',
      'Equivalence Principle',
      'Gravitational Time Dilation',
      'GPS and Relativity'
    ],
    estimatedTime: '60 min'
  },
  {
    id: 'black-holes',
    title: 'Black Holes and Extreme Gravity',
    category: 'Modern Physics',
    difficulty: 'Advanced',
    topics: [
      'Event Horizons',
      'Schwarzschild Radius',
      'Hawking Radiation',
      'Gravitational Lensing'
    ],
    estimatedTime: '55 min'
  }
];

const LessonCard = ({ lesson, index }) => {
  const navigate = useNavigate();
  const difficultyColor = {
    'Beginner': 'success',
    'Intermediate': 'warning',
    'Advanced': 'error'
  };
  
  return (
    <motion.div
      initial={{ opacity: 0, x: -50 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.5, delay: index * 0.1 }}
    >
      <Card 
        sx={{ 
          mb: 2,
          cursor: 'pointer',
          transition: 'all 0.3s',
          '&:hover': {
            transform: 'translateX(10px)',
            boxShadow: 3
          }
        }}
        onClick={() => navigate(`/learn/lesson/${lesson.id}`)}
      >
        <CardContent>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start' }}>
            <Box sx={{ flex: 1 }}>
              <Typography variant="h6" gutterBottom>
                {lesson.title}
              </Typography>
              <Box sx={{ display: 'flex', gap: 1, mb: 1 }}>
                <Chip 
                  label={lesson.category} 
                  size="small" 
                  variant="outlined"
                />
                <Chip 
                  label={lesson.difficulty} 
                  size="small" 
                  color={difficultyColor[lesson.difficulty]}
                />
                <Chip 
                  label={lesson.estimatedTime} 
                  size="small" 
                  icon={<School />}
                />
              </Box>
              <List dense>
                {lesson.topics.slice(0, 3).map((topic, i) => (
                  <ListItem key={i} sx={{ py: 0 }}>
                    <ListItemText 
                      primary={`• ${topic}`}
                      primaryTypographyProps={{ variant: 'body2' }}
                    />
                  </ListItem>
                ))}
              </List>
            </Box>
            <ArrowForward sx={{ mt: 2, color: 'primary.main' }} />
          </Box>
        </CardContent>
      </Card>
    </motion.div>
  );
};

const LearnMenu = () => {
  const [selectedCategory, setSelectedCategory] = useState(0);
  const categories = ['All', 'Classical Mechanics', 'Modern Physics'];
  
  const filteredLessons = selectedCategory === 0 
    ? lessons 
    : lessons.filter(l => l.category === categories[selectedCategory]);
  
  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Typography variant="h3" gutterBottom align="center">
        📚 Learn Gravity
      </Typography>
      <Typography variant="h6" color="text.secondary" align="center" sx={{ mb: 6 }}>
        Master gravitational physics from basics to advanced concepts
      </Typography>
      
      <Grid container spacing={4}>
        <Grid item xs={12} md={3}>
          <Paper sx={{ p: 2, position: 'sticky', top: 80 }}>
            <Typography variant="h6" gutterBottom>
              Categories
            </Typography>
            <Tabs
              orientation="vertical"
              value={selectedCategory}
              onChange={(e, v) => setSelectedCategory(v)}
              sx={{ borderRight: 1, borderColor: 'divider' }}
            >
              {categories.map((cat, i) => (
                <Tab key={i} label={cat} />
              ))}
            </Tabs>
            
            <Divider sx={{ my: 3 }} />
            
            <Typography variant="h6" gutterBottom>
              Learning Tools
            </Typography>
            <List>
              <ListItemButton component={Link} to="/learn/calculator">
                <Calculate sx={{ mr: 2 }} />
                <ListItemText primary="Physics Calculator" />
              </ListItemButton>
              <ListItemButton component={Link} to="/learn/quiz">
                <Quiz sx={{ mr: 2 }} />
                <ListItemText primary="Practice Quiz" />
              </ListItemButton>
              <ListItemButton component={Link} to="/learn/videos">
                <VideoLibrary sx={{ mr: 2 }} />
                <ListItemText primary="Video Library" />
              </ListItemButton>
            </List>
          </Paper>
        </Grid>
        
        <Grid item xs={12} md={9}>
          {filteredLessons.map((lesson, index) => (
            <LessonCard key={lesson.id} lesson={lesson} index={index} />
          ))}
        </Grid>
      </Grid>
    </Container>
  );
};

const Learn = () => {
  return (
    <Routes>
      <Route path="/" element={<LearnMenu />} />
      <Route path="/lesson/:lessonId" element={<LessonViewer />} />
      <Route path="/calculator" element={<InteractiveDemo />} />
      <Route path="/quiz" element={<QuizComponent />} />
    </Routes>
  );
};

export default Learn;