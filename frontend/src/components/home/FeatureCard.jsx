import React from 'react';
import { Card, CardContent, Box, Typography } from '@mui/material';
import { motion } from 'framer-motion';

const FeatureCard = ({ icon, title, description, color }) => {
  return (
    <motion.div whileHover={{ y: -10 }} transition={{ type: 'spring', stiffness: 300 }}>
      <Card
        sx={{
          height: '100%',
          background: 'rgba(255, 255, 255, 0.05)',
          backdropFilter: 'blur(10px)',
          border: '1px solid rgba(255, 255, 255, 0.1)',
          transition: 'all 0.3s',
          '&:hover': {
            border: `1px solid ${color}`,
            boxShadow: `0 8px 32px ${color}40`
          }
        }}
      >
        <CardContent>
          <Box
            sx={{
              width: 60,
              height: 60,
              borderRadius: 2,
              background: `linear-gradient(135deg, ${color}40, ${color}20)`,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              mb: 2,
              color: color
            }}
          >
            {icon}
          </Box>
          
          <Typography variant="h6" gutterBottom>
            {title}
          </Typography>
          
          <Typography variant="body2" color="text.secondary">
            {description}
          </Typography>
        </CardContent>
      </Card>
    </motion.div>
  );
};

export default FeatureCard;