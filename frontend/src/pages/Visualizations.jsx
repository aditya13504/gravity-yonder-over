import React, { useState } from 'react';
import {
  Box,
  Container,
  Grid,
  Card,
  CardContent,
  Typography,
  ToggleButtonGroup,
  ToggleButton,
  Slider,
  Paper,
  Divider
} from '@mui/material';
import { motion } from 'framer-motion';
import GravityFieldVisualizer from '../components/visualizations/GravityFieldVisualizer';
import OrbitTrackerVisualizer from '../components/visualizations/OrbitTrackerVisualizer';
import RelativisticEffectsVisualizer from '../components/visualizations/RelativisticEffectsVisualizer';
import TidalForcesVisualizer from '../components/visualizations/TidalForcesVisualizer';

const Visualizations = () => {
  const [activeViz, setActiveViz] = useState('gravity-field');
  const [parameters, setParameters] = useState({
    mass: 1,
    distance: 10,
    velocity: 30,
    time: 0
  });
  
  const visualizations = {
    'gravity-field': {
      title: 'Gravitational Field',
      component: GravityFieldVisualizer,
      description: 'Visualize how mass curves spacetime and creates gravitational fields',
      controls: ['mass', 'distance']
    },
    'orbit-tracker': {
      title: 'Orbital Mechanics',
      component: OrbitTrackerVisualizer,
      description: 'Track orbital paths and understand Kepler\'s laws in action',
      controls: ['velocity', 'distance']
    },
    'relativistic': {
      title: 'Relativistic Effects',
      component: RelativisticEffectsVisualizer,
      description: 'See how extreme gravity affects time and space',
      controls: ['mass', 'distance', 'time']
    },
    'tidal-forces': {
      title: 'Tidal Forces',
      component: TidalForcesVisualizer,
      description: 'Understand how gravity creates tides and can tear objects apart',
      controls: ['mass', 'distance']
    }
  };
  
  const ActiveVisualization = visualizations[activeViz].component;
  
  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      <Typography variant="h3" gutterBottom align="center">
        📊 Gravity Visualizations
      </Typography>
      <Typography variant="h6" color="text.secondary" align="center" sx={{ mb: 6 }}>
        Interactive visualizations to understand gravitational phenomena
      </Typography>
      
      <Grid container spacing={3}>
        {/* Visualization Selection */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <ToggleButtonGroup
              value={activeViz}
              exclusive
              onChange={(e, v) => v && setActiveViz(v)}
              fullWidth
            >
              {Object.entries(visualizations).map(([key, viz]) => (
                <ToggleButton key={key} value={key}>
                  {viz.title}
                </ToggleButton>
              ))}
            </ToggleButtonGroup>
          </Paper>
        </Grid>
        
        {/* Main Visualization */}
        <Grid item xs={12} lg={9}>
          <motion.div
            key={activeViz}
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5 }}
          >
            <Card sx={{ height: 600 }}>
              <ActiveVisualization parameters={parameters} />
            </Card>
          </motion.div>
        </Grid>
        
        {/* Controls */}
        <Grid item xs={12} lg={3}>
          <Paper sx={{ p: 3, position: 'sticky', top: 80 }}>
            <Typography variant="h6" gutterBottom>
              {visualizations[activeViz].title}
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
              {visualizations[activeViz].description}
            </Typography>
            
            <Divider sx={{ my: 2 }} />
            
            <Typography variant="subtitle1" gutterBottom>
              Parameters
            </Typography>
            
            {visualizations[activeViz].controls.includes('mass') && (
              <Box sx={{ mb: 3 }}>
                <Typography gutterBottom>
                  Mass: {parameters.mass} Solar Masses
                </Typography>
                <Slider
                  value={parameters.mass}
                  onChange={(e, v) => setParameters({ ...parameters, mass: v })}
                  min={0.1}
                  max={10}
                  step={0.1}
                />
              </Box>
            )}
            
            {visualizations[activeViz].controls.includes('distance') && (
              <Box sx={{ mb: 3 }}>
                <Typography gutterBottom>
                  Distance: {parameters.distance} AU
                </Typography>
                <Slider
                  value={parameters.distance}
                  onChange={(e, v) => setParameters({ ...parameters, distance: v })}
                  min={1}
                  max={50}
                />
              </Box>
            )}
            
            {visualizations[activeViz].controls.includes('velocity') && (
              <Box sx={{ mb: 3 }}>
                <Typography gutterBottom>
                  Velocity: {parameters.velocity} km/s
                </Typography>
                <Slider
                  value={parameters.velocity}
                  onChange={(e, v) => setParameters({ ...parameters, velocity: v })}
                  min={0}
                  max={100}
                />
              </Box>
            )}
            
            {visualizations[activeViz].controls.includes('time') && (
              <Box sx={{ mb: 3 }}>
                <Typography gutterBottom>
                  Time: {parameters.time} years
                </Typography>
                <Slider
                  value={parameters.time}
                  onChange={(e, v) => setParameters({ ...parameters, time: v })}
                  min={0}
                  max={100}
                />
              </Box>
            )}
            
            <Divider sx={{ my: 2 }} />
            
            <Typography variant="subtitle2" gutterBottom>
              Quick Facts
            </Typography>
            <Typography variant="body2" color="text.secondary">
              • Earth's escape velocity: 11.2 km/s
            </Typography>
            <Typography variant="body2" color="text.secondary">
              • 1 AU = 149.6 million km
            </Typography>
            <Typography variant="body2" color="text.secondary">
              • Sun's mass = 333,000 Earth masses
            </Typography>
          </Paper>
        </Grid>
      </Grid>
    </Container>
  );
};

export default Visualizations;