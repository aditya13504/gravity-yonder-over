import React from 'react';
import { Link } from 'react-router-dom';
import { 
  Box, 
  Typography, 
  Button, 
  Grid, 
  Card, 
  CardContent, 
  CardMedia,
  Container,
  Chip
} from '@mui/material';
import { motion } from 'framer-motion';
import { 
  RocketLaunch, 
  Science, 
  School, 
  EmojiEvents,
  Speed,
  Devices
} from '@mui/icons-material';
import HeroSection from '../components/home/HeroSection';
import FeatureCard from '../components/home/FeatureCard';
import StatsDisplay from '../components/home/StatsDisplay';

const Home = () => {
  const features = [
    {
      icon: <RocketLaunch fontSize="large" />,
      title: '5 Interactive Games',
      description: 'Learn gravity through hands-on mini-games, from apple drops to black hole navigation.',
      color: '#4169E1'
    },
    {
      icon: <Science fontSize="large" />,
      title: 'Physics Sandbox',
      description: 'Create your own solar systems and experiment with gravitational interactions.',
      color: '#FDB813'
    },
    {
      icon: <School fontSize="large" />,
      title: 'Educational Content',
      description: 'Structured lessons from Newton to Einstein, perfect for students and educators.',
      color: '#32CD32'
    },
    {
      icon: <Speed fontSize="large" />,
      title: 'Real-time Simulations',
      description: 'GPU-accelerated physics for smooth, accurate gravitational modeling.',
      color: '#FF6347'
    },
    {
      icon: <Devices fontSize="large" />,
      title: 'Cross-Platform',
      description: 'Works on any device - desktop, tablet, or mobile. No installation required.',
      color: '#9370DB'
    },
    {
      icon: <EmojiEvents fontSize="large" />,
      title: 'Achievement System',
      description: 'Track your progress, earn badges, and compete with friends.',
      color: '#FFD700'
    }
  ];

  return (
    <Box>
      <HeroSection />
      
      <Container maxWidth="lg" sx={{ py: 8 }}>
        {/* Features Section */}
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
        >
          <Typography variant="h3" align="center" gutterBottom>
            Explore the Universe of Gravity
          </Typography>
          <Typography variant="h6" align="center" color="text.secondary" sx={{ mb: 6 }}>
            Master the fundamental force that shapes our cosmos through interactive learning
          </Typography>
          
          <Grid container spacing={4}>
            {features.map((feature, index) => (
              <Grid item xs={12} md={4} key={index}>
                <motion.div
                  initial={{ opacity: 0, y: 30 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
                  viewport={{ once: true }}
                >
                  <FeatureCard {...feature} />
                </motion.div>
              </Grid>
            ))}
          </Grid>
        </motion.div>
        
        {/* Learning Path Section */}
        <Box sx={{ mt: 10 }}>
          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            transition={{ duration: 1 }}
            viewport={{ once: true }}
          >
            <Typography variant="h4" align="center" gutterBottom>
              Your Learning Journey
            </Typography>
            
            <Grid container spacing={3} sx={{ mt: 4 }}>
              {[
                { level: 'Beginner', topics: ['Newton\'s Laws', 'Free Fall', 'Basic Orbits'] },
                { level: 'Intermediate', topics: ['Kepler\'s Laws', 'Escape Velocity', 'Tidal Forces'] },
                { level: 'Advanced', topics: ['General Relativity', 'Black Holes', 'Gravitational Waves'] }
              ].map((path, index) => (
                <Grid item xs={12} md={4} key={index}>
                  <Card 
                    sx={{ 
                      height: '100%',
                      background: 'linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02))',
                      backdropFilter: 'blur(10px)',
                      border: '1px solid rgba(255,255,255,0.1)'
                    }}
                  >
                    <CardContent>
                      <Typography variant="h5" gutterBottom>
                        {path.level}
                      </Typography>
                      {path.topics.map((topic, i) => (
                        <Chip 
                          key={i} 
                          label={topic} 
                          size="small" 
                          sx={{ m: 0.5 }}
                          variant="outlined"
                        />
                      ))}
                    </CardContent>
                  </Card>
                </Grid>
              ))}
            </Grid>
          </motion.div>
        </Box>
        
        {/* Stats Section */}
        <StatsDisplay />
        
        {/* CTA Section */}
        <Box sx={{ mt: 10, textAlign: 'center' }}>
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            whileInView={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5 }}
            viewport={{ once: true }}
          >
            <Typography variant="h4" gutterBottom>
              Ready to Start Your Cosmic Journey?
            </Typography>
            <Box sx={{ mt: 4, display: 'flex', gap: 2, justifyContent: 'center' }}>
              <Button
                component={Link}
                to="/games"
                variant="contained"
                size="large"
                sx={{ px: 4, py: 1.5 }}
              >
                Play Games
              </Button>
              <Button
                component={Link}
                to="/learn"
                variant="outlined"
                size="large"
                sx={{ px: 4, py: 1.5 }}
              >
                Start Learning
              </Button>
            </Box>
          </motion.div>
        </Box>
      </Container>
    </Box>
  );
};

export default Home;