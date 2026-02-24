// Social and brand icons
import { FaXTwitter, FaGithub, FaLinkedin, FaFacebook, FaNewspaper, FaCopy,FaBook } from "react-icons/fa6";
import { IoMailOutline } from "react-icons/io5";
import { FaGraduationCap } from "react-icons/fa";

// UI and navigation icons
import { 
  FiArrowLeft, 
  FiArrowRight, 
  FiMoon, 
  FiSun, 
  FiX, 
  FiCheck, 
  FiChevronRight, 
  FiCircle, 
  FiShare2, 
  FiLink, 
  FiMessageCircle,
  FiMenu
} from "react-icons/fi";

// Location and layout icons
import { MdLocationPin } from "react-icons/md";
import { TbLogs, TbLayoutSidebar } from "react-icons/tb";

// ============================================
// Icon exports
// ============================================

// Social media icons
export const Twitter = FaXTwitter;
export const Mail = IoMailOutline;
export const Github = FaGithub;
export const Linkedin = FaLinkedin;
export const Facebook = FaFacebook;

// Content icons
export const GraduationCap = FaGraduationCap;
export const MapPin = MdLocationPin;
export const News = FaNewspaper;
export const Blogs = FaBook;
export const Copy = FaCopy;

// Navigation icons
export const ArrowLeft = FiArrowLeft;
export const ArrowRight = FiArrowRight;
export const ChevronRight = FiChevronRight;

// UI state icons
export const Moon = FiMoon;
export const Sun = FiSun;
export const XIcon = FiX;
export const X = FiX;
export const Menu = FiMenu;
export const Check = FiCheck;
export const Circle = FiCircle;

// Action icons
export const Share2 = FiShare2;
export const Link = FiLink;
export const MessageCircle = FiMessageCircle;
export const PanelLeft = TbLayoutSidebar;

// Aliases matching previous/lucide names used across the codebase
export const CheckIcon = FiCheck;
export const ChevronRightIcon = FiChevronRight;
export const CircleIcon = FiCircle;
export const PanelLeftIcon = TbLayoutSidebar;
export const LinkIcon = FiLink;

// Also provide common "*Icon" aliases for social/content icons
export const TwitterIcon = FaXTwitter;
export const MailIcon = IoMailOutline;
export const GithubIcon = FaGithub;
export const LinkedinIcon = FaLinkedin;
export const GraduationCapIcon = FaGraduationCap;
export const NewsIcon = FaNewspaper;
export const LogsIcon = TbLogs;
export const CopyIcon = FaCopy;