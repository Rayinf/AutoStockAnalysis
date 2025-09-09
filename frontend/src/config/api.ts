// API配置文件
// 根据环境自动选择API基础URL

// 开发环境API地址
const DEV_API_BASE_URL = 'http://localhost:8000';

// 生产环境API地址 (Railway部署)
const PROD_API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'https://autostockanalysis-production.up.railway.app';

// GitHub Pages环境API地址
const GITHUB_PAGES_API_URL = import.meta.env.VITE_API_BASE_URL || 'https://autostockanalysis-production.up.railway.app';

/**
 * 获取API基础URL
 * 根据当前环境自动选择合适的API地址
 */
export const getApiBaseUrl = (): string => {
  // 检查是否在开发环境
  if (import.meta.env.DEV) {
    return DEV_API_BASE_URL;
  }
  
  // 检查是否在GitHub Pages环境
  if (window.location.hostname.includes('github.io')) {
    return GITHUB_PAGES_API_URL;
  }
  
  // 生产环境
  return PROD_API_BASE_URL;
};

/**
 * API端点配置
 */
export const API_ENDPOINTS = {
  // 股票分析相关
  ANALYZE: '/analyze',
  PREDICT: '/predict',
  BACKTEST: '/backtest',
  
  // 数据获取相关
  STOCK_DATA: '/stock/data',
  STOCK_INFO: '/stock/info',
  MARKET_DATA: '/market/data',
  
  // 用户相关
  USER_SETTINGS: '/user/settings',
  CHAT_HISTORY: '/chat/history',
} as const;

/**
 * 构建完整的API URL
 * @param endpoint API端点
 * @returns 完整的API URL
 */
export const buildApiUrl = (endpoint: string): string => {
  const baseUrl = getApiBaseUrl();
  return `${baseUrl}${endpoint}`;
};

/**
 * API请求配置
 */
export const API_CONFIG = {
  timeout: 30000, // 30秒超时
  headers: {
    'Content-Type': 'application/json',
  },
};

/**
 * 检查API服务是否可用
 * @returns Promise<boolean>
 */
export const checkApiHealth = async (): Promise<boolean> => {
  try {
    const response = await fetch(`${getApiBaseUrl()}/health`, {
      method: 'GET',
      ...API_CONFIG,
    });
    return response.ok;
  } catch (error) {
    console.error('API健康检查失败:', error);
    return false;
  }
};