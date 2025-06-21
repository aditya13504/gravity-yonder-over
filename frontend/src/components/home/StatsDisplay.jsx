import React, { useState, useEffect } from 'react';
import { Box, Grid, Typography, Paper } from '@mui/material';
import { motion } from 'framer-motion';
import CountUp from 'react-countup';

const StatsDisplay = () => {
  const [isVisible, setIsVisible] = useState(false);
  
  const stats = [
    { label: 'Active Learners', value: 10000, suffix: '+' },
    { label: 'Simulations Run', value: 250000, suffix: '' },
    { label: 'Concepts Covered', value: 50, suffix: '+' },
    { label: 'Success Rate', value: 94, suffix: '%' }
  ];
  
  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true);
        }
      },
      { threshold: 0.1 }
    );
    
    const element = document.getElementById('stats-section');
    if (element) observer.observe(element);
    
    return () => {
      if (element) observer.unobserve(element);
    };
  }, []);
  
  return (
    <Box id="stats-section" sx={{ mt: 10 }}>
      <Typography variant="h4" align="center" gutterBottom>
        Join Thousands Learning Gravity
      </Typography>
      
      <Grid container spacing={3} sx={{ mt: 4 }}>
        {stats.map((stat, index) => (
          <Grid item xs={6} md={3} key={index}>
            <motion.div
              initial={{ opacity: 0, y: 50 }}
              animate={isVisible ? { opacity: 1, y: 0 } : {}}
              transition={{ duration: 0.5, delay: index * 0.1 }}
            >
              <Paper
                sx={{
                  p: 3,
                  textAlign: 'center',
                  background: 'rgba(255, 255, 255, 0.05)',
                  backdropFilter: 'blur(10px)',
                  border: '1px solid rgba(255, 255, 255, 0.1)'
                }}
              >
                <Typography variant="h3" sx={{ fontWeight: 700 }}>
                  {isVisible && (
                    <CountUp
                      start={0}
                      end={stat.value}
                      duration={2}
                      separator=","
                      suffix={stat.suffix}
                    />
                  )}
                </Typography>
                <Typography variant="body1" color="text.secondary">
                  {stat.label}
                </Typography>
              </Paper>
            </motion.div>
          </Grid>
        ))}
      </Grid>
    </Box>
  );
};

export default StatsDisplay;