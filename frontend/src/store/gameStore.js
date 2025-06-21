import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export const useGameStore = create(
  persist(
    (set, get) => ({
      // Player stats
      score: 0,
      level: 1,
      achievements: [],
      unlockedGames: ['AppleDrop'],
        // Game progress
      gameProgress: {
        AppleDrop: { completed: false, bestScore: 0 },
        OrbitalSlingshot: { completed: false, bestScore: 0 },
        EscapeVelocity: { completed: false, bestScore: 0 },
        BlackHoleNavigator: { completed: false, bestScore: 0 },
        LagrangeExplorer: { completed: false, bestScore: 0 },
        WormholeNavigator: { completed: false, bestScore: 0 },
      },
      
      // Actions
      addScore: (points) => set((state) => ({ 
        score: state.score + points,
        level: Math.floor((state.score + points) / 100) + 1
      })),
      
      unlockGame: (gameName) => set((state) => ({
        unlockedGames: [...new Set([...state.unlockedGames, gameName])]
      })),
      
      updateGameProgress: (gameName, progress) => set((state) => ({
        gameProgress: {
          ...state.gameProgress,
          [gameName]: {
            ...state.gameProgress[gameName],
            ...progress,
          }
        }
      })),
      
      addAchievement: (achievement) => set((state) => ({
        achievements: [...new Set([...state.achievements, achievement])]
      })),
      
      resetProgress: () => set({
        score: 0,
        level: 1,
        achievements: [],
        unlockedGames: ['AppleDrop'],
        gameProgress: {
          AppleDrop: { completed: false, bestScore: 0 },
          OrbitalSlingshot: { completed: false, bestScore: 0 },
          EscapeVelocity: { completed: false, bestScore: 0 },
          BlackHoleNavigator: { completed: false, bestScore: 0 },
          LagrangeExplorer: { completed: false, bestScore: 0 },
        },
      }),
    }),
    {
      name: 'gravity-yonder-game-state',
    }
  )
);