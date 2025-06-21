import React from 'react';
import { Routes, Route, Link, useNavigate } from 'react-router-dom';
import { 
  Box, 
  Grid, 
  Card, 
  CardContent, 
  CardActions,
  Typography, 
  Button,
  Chip,
  LinearProgress,
  Container
} from '@mui/material';
import { motion } from 'framer-motion';
import { Lock, PlayArrow, Star } from '@mui/icons-material';
import { useGameStore } from '../store/gameStore';
import AppleDropGame from '../components/games/AppleDrop';
import OrbitalSlingshotGame from '../components/games/OrbitalSlingshot';
import EscapeVelocityGame from '../components/games/EscapeVelocity';
import BlackHoleNavigatorGame from '../components/games/BlackHoleNavigator';
import LagrangeExplorerGame from '../components/games/LagrangeExplorer';
import WormholeNavigatorGame from '../components/games/WormholeNavigator';

const GameCard = ({ game, index, isUnlocked, progress }) => {
  const navigate = useNavigate();
  
  return (
    <motion.div
      initial={{ opacity: 0, y: 50 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: index * 0.1 }}
    >
      <Card 
        sx={{ 
          height: '100%',
          opacity: isUnlocked ? 1 : 0.6,
          transition: 'all 0.3s',
          '&:hover': {
            transform: isUnlocked ? 'scale(1.05)' : 'none',
            boxShadow: isUnlocked ? '0 8px 32px rgba(0,0,0,0.3)' : 'none'
          }
        }}
      >
        <Box
          sx={{
            height: 200,
            background: `linear-gradient(135deg, ${game.color1}, ${game.color2})`,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            position: 'relative'
          }}
        >
          <Typography variant="h1" sx={{ opacity: 0.2 }}>
            {game.icon}
          </Typography>
          {!isUnlocked && (
            <Box
              sx={{
                position: 'absolute',
                inset: 0,
                backgroundColor: 'rgba(0,0,0,0.7)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center'
              }}
            >
              <Lock sx={{ fontSize: 60 }} />
            </Box>
          )}
        </Box>
        
        <CardContent>
          <Typography variant="h5" gutterBottom>
            {game.name}
          </Typography>
          <Chip 
            label={game.difficulty} 
            size="small" 
            color={
              game.difficulty === 'Beginner' ? 'success' :
              game.difficulty === 'Intermediate' ? 'warning' :
              'error'
            }
            sx={{ mb: 2 }}
          />
          <Typography variant="body2" color="text.secondary">
            {game.description}
          </Typography>
          
          {isUnlocked && progress && (
            <Box sx={{ mt: 2 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                <Typography variant="body2">Progress</Typography>
                <Typography variant="body2">{progress.bestScore} pts</Typography>
              </Box>
              <LinearProgress 
                variant="determinate" 
                value={Math.min((progress.bestScore / game.maxScore) * 100, 100)} 
              />
            </Box>
          )}
        </CardContent>
        
        <CardActions>
          <Button
            fullWidth
            variant={isUnlocked ? "contained" : "outlined"}
            disabled={!isUnlocked}
            startIcon={isUnlocked ? <PlayArrow /> : <Lock />}
            onClick={() => navigate(`/games/${game.path}`)}
          >
            {isUnlocked ? 'Play' : 'Locked'}
          </Button>
        </CardActions>
      </Card>
    </motion.div>
  );
};

const GamesMenu = () => {
  const { unlockedGames, gameProgress } = useGameStore();
  
  const games = [
    {
      name: 'Apple Drop',
      path: 'apple-drop',
      icon: '🍎',
      description: 'Experience Newton\'s discovery! Drop apples and observe how gravity affects their motion.',
      difficulty: 'Beginner',
      color1: '#ff6b6b',
      color2: '#ee5a6f',
      maxScore: 100
    },
    {
      name: 'Orbital Slingshot',
      path: 'orbital-slingshot',
      icon: '🛸',
      description: 'Master gravitational assists to accelerate spacecraft and reach distant planets.',
      difficulty: 'Intermediate',
      color1: '#4ecdc4',
      color2: '#44a08d',
      maxScore: 200
    },
    {
      name: 'Escape Velocity',
      path: 'escape-velocity',
      icon: '🚀',
      description: 'Launch rockets and calculate the exact speed needed to escape planetary gravity.',
      difficulty: 'Intermediate',
      color1: '#a8e063',
      color2: '#56ab2f',
      maxScore: 300
    },
    {
      name: 'Black Hole Navigator',
      path: 'black-hole-navigator',
      icon: '⚫',
      description: 'Navigate spacecraft near black holes without crossing the event horizon.',
      difficulty: 'Advanced',
      color1: '#654ea3',
      color2: '#eaafc8',
      maxScore: 400
    },    {
      name: 'Lagrange Explorer',
      path: 'lagrange-explorer',
      icon: '🌐',
      description: 'Find and utilize gravitational equilibrium points between celestial bodies.',
      difficulty: 'Expert',
      color1: '#f093fb',
      color2: '#f5576c',
      maxScore: 500
    },
    {
      name: 'Wormhole Navigator',
      path: 'wormhole-navigator',
      icon: '🌌',
      description: 'Traverse Einstein-Rosen bridges and explore exotic spacetime geometries.',
      difficulty: 'Master',
      color1: '#8A2BE2',
      color2: '#4B0082',
      maxScore: 600
    }
  ];
  
  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Typography variant="h3" gutterBottom align="center">
        🎮 Gravity Games
      </Typography>
      <Typography variant="h6" color="text.secondary" align="center" sx={{ mb: 6 }}>
        Learn physics through interactive challenges
      </Typography>
      
      <Grid container spacing={4}>
        {games.map((game, index) => (
          <Grid item xs={12} md={6} lg={4} key={game.path}>
            <GameCard
              game={game}
              index={index}
              isUnlocked={unlockedGames.includes(game.name.replace(' ', ''))}
              progress={gameProgress[game.name.replace(' ', '')]}
            />
          </Grid>
        ))}
      </Grid>
    </Container>
  );
};

const Games = () => {
  return (
    <Routes>
      <Route path="/" element={<GamesMenu />} />
      <Route path="/apple-drop" element={<AppleDropGame />} />
      <Route path="/orbital-slingshot" element={<OrbitalSlingshotGame />} />
      <Route path="/escape-velocity" element={<EscapeVelocityGame />} />
      <Route path="/black-hole-navigator" element={<BlackHoleNavigatorGame />} />
      <Route path="/lagrange-explorer" element={<LagrangeExplorerGame />} />
      <Route path="/wormhole-navigator" element={<WormholeNavigatorGame />} />
    </Routes>
  );
};

export default Games;