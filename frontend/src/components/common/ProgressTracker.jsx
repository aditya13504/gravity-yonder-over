import React from 'react';
import {
  Box,
  LinearProgress,
  Typography,
  Chip,
  Tooltip,
  Card,
  CardContent
} from '@mui/material';
import { motion } from 'framer-motion';
import { Star, Trophy, EmojiEvents } from '@mui/icons-material';
import { useGameStore } from '../../store/gameStore';

const ProgressTracker = ({ compact = false }) => {
  const { score, level, achievements, gameProgress } = useGameStore();
  
  const levelProgress = (score % 100);
  const nextLevelScore = Math.ceil(score / 100) * 100;
  
  const recentAchievements = achievements.slice(-3);
  
  if (compact) {
    return (
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
        <Chip
          icon={<Trophy />}
          label={`Level ${level}`}
          color="primary"
          size="small"
        />
        <Box sx={{ flex: 1 }}>
          <LinearProgress
            variant="determinate"
            value={levelProgress}
            sx={{ height: 6, borderRadius: 3 }}
          />
        </Box>
        <Typography variant="body2">
          {score} pts
        </Typography>
      </Box>
    );
  }
  
  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Progress Tracker
        </Typography>
        
        {/* Level Progress */}
        <Box sx={{ mb: 3 }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
            <Typography variant="body2">
              Level {level}
            </Typography>
            <Typography variant="body2">
              {score} / {nextLevelScore} pts
            </Typography>
          </Box>
          <LinearProgress
            variant="determinate"
            value={levelProgress}
            sx={{ height: 10, borderRadius: 5 }}
          />
        </Box>
        
        {/* Game Progress Summary */}
        <Typography variant="subtitle2" gutterBottom>
          Game Completion
        </Typography>
        <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', mb: 3 }}>
          {Object.entries(gameProgress).map(([game, progress]) => (
            <Tooltip key={game} title={`Best Score: ${progress.bestScore}`}>
              <Chip
                size="small"
                icon={progress.completed ? <Star /> : null}
                label={game.replace(/([A-Z])/g, ' $1').trim()}
                color={progress.completed ? 'success' : 'default'}
                variant={progress.completed ? 'filled' : 'outlined'}
              />
            </Tooltip>
          ))}
        </Box>
        
        {/* Recent Achievements */}
        {recentAchievements.length > 0 && (
          <>
            <Typography variant="subtitle2" gutterBottom>
              Recent Achievements
            </Typography>
            <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
              {recentAchievements.map((achievement, index) => (
                <motion.div
                  key={achievement}
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  transition={{ delay: index * 0.1 }}
                >
                  <Chip
                    size="small"
                    icon={<EmojiEvents />}
                    label={achievement}
                    color="warning"
                  />
                </motion.div>
              ))}
            </Box>
          </>
        )}
      </CardContent>
    </Card>
  );
};

export default ProgressTracker;