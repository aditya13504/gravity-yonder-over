import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import {
  AppBar,
  Toolbar,
  Typography,
  Button,
  IconButton,
  Drawer,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Box,
  useMediaQuery,
  useTheme,
  Badge,
} from '@mui/material';
import {
  Menu as MenuIcon,
  Home as HomeIcon,
  SportsEsports as GamesIcon,
  Science as SandboxIcon,
  School as LearnIcon,
  BarChart as VisualizationIcon,
  EmojiEvents as TrophyIcon,
} from '@mui/icons-material';
import { motion } from 'framer-motion';
import { useGameStore } from '../../store/gameStore';

const Navigation = () => {
  const [drawerOpen, setDrawerOpen] = useState(false);
  const location = useLocation();
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const { score, level } = useGameStore();

  const navItems = [
    { text: 'Home', icon: <HomeIcon />, path: '/' },
    { text: 'Games', icon: <GamesIcon />, path: '/games' },
    { text: 'Sandbox', icon: <SandboxIcon />, path: '/sandbox' },
    { text: 'Learn', icon: <LearnIcon />, path: '/learn' },
    { text: 'Visualizations', icon: <VisualizationIcon />, path: '/visualizations' },
  ];

  const isActive = (path) => location.pathname.startsWith(path) || 
    (path === '/' && location.pathname === '/');

  return (
    <>
      <AppBar position="fixed" sx={{ zIndex: theme.zIndex.drawer + 1 }}>
        <Toolbar>
          {isMobile && (
            <IconButton
              color="inherit"
              edge="start"
              onClick={() => setDrawerOpen(true)}
              sx={{ mr: 2 }}
            >
              <MenuIcon />
            </IconButton>
          )}
          
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            <motion.div
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
            >
              🌌 Gravity Yonder Over
            </motion.div>
          </Typography>

          {!isMobile && (
            <Box sx={{ display: 'flex', gap: 2 }}>
              {navItems.map((item) => (
                <Button
                  key={item.path}
                  component={Link}
                  to={item.path}
                  color="inherit"
                  startIcon={item.icon}
                  sx={{
                    borderBottom: isActive(item.path) ? '2px solid white' : 'none',
                    borderRadius: 0,
                  }}
                >
                  {item.text}
                </Button>
              ))}
            </Box>
          )}

          <Box sx={{ display: 'flex', alignItems: 'center', ml: 2 }}>
            <Badge badgeContent={level} color="secondary">
              <TrophyIcon />
            </Badge>
            <Typography variant="body2" sx={{ ml: 1 }}>
              {score} pts
            </Typography>
          </Box>
        </Toolbar>
      </AppBar>

      <Drawer
        anchor="left"
        open={drawerOpen}
        onClose={() => setDrawerOpen(false)}
      >
        <Box sx={{ width: 250, pt: 8 }}>
          <List>
            {navItems.map((item) => (
              <ListItem
                key={item.path}
                component={Link}
                to={item.path}
                onClick={() => setDrawerOpen(false)}
                sx={{
                  backgroundColor: isActive(item.path) ? 'action.selected' : 'transparent',
                }}
              >
                <ListItemIcon>{item.icon}</ListItemIcon>
                <ListItemText primary={item.text} />
              </ListItem>
            ))}
          </List>
        </Box>
      </Drawer>

      <Toolbar /> {/* Spacer */}
    </>
  );
};

export default Navigation;