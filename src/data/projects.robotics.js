import robotYoloImg from "../assets/robot-yolo.png"; 
import armImg from "../assets/Ensamble_Brazo.png"; 
import correcaminos from "../assets/correcaminos.jpeg"; 

const ROBOTICS_PROJECTS = [
  {
    title: "Autonomous Mobile Robot with Vision-Based Safety (YOLOv5)",
    meta: "Arduino, Raspberry Pi 4, embedded control, CV integration",
    image: robotYoloImg,
    alt: "YOLOv5 robot preview",
    bullets: [
      "Built an autonomous mobile robot integrating mechanical, electronic, and software components for real-world navigation tasks.",
      "Implemented embedded control and communication pipelines using Arduino and Raspberry Pi 4 to support autonomous operation and system coordination.",
      "Developed a YOLOv5-based vision module to detect restricted-area signs and prevent unauthorized crossings, achieving high-confidence detections in testing."
    ],
    links: [
    
      { label: "YouTube Demo", href: "https://www.youtube.com/watch?v=JE850tpixUE" },
      { label: "IEEE Publication", href: "https://ieeexplore.ieee.org/document/10405395" },
    ],
  },
  {
    title: "3DOF Pneumatic Robotic Arm (Remote Control)",
    meta: "Pneumatics, mechatronics integration, remote actuation",
    image: armImg,
    alt: "3DOF pneumatic arm preview",
    bullets: [
      "Designed and assembled a 3DOF pneumatic robotic arm for object manipulation experiments.",
      "Integrated remote-control actuation and safety constraints for controlled motion and repeatable demonstrations.",
      "Validated pick-and-place style movements with stable response under pneumatic variability."
    ],
    links: [
      { label: "Demo", href: "https://drive.google.com/file/d/1YkjZtpVBqmxntnnmAefZ0UnL5WFi-1kV/view?usp=sharing"}
    ],
  },

  {
  title: "Animatronic RoadRunner",
  meta: "Mechatronics design, 3D printing, embedded control, Robotics",
  image: correcaminos,
  alt: "Animatronic RoadRunner robot preview",
  bullets: [
    "Designed and fabricated a custom animatronic robot using 3D-printed structural components and a mechatronic actuation system.",
    "Implemented embedded control logic to synchronize locomotion with dynamic LED eye activation during movement.",
    "Integrated a rear proximity sensor to detect obstacles and trigger adaptive behavior, including characteristic RoadRunner sound effects."
  ],
  links: [
    { 
      label: "Demo", 
      href: "https://drive.google.com/file/d/1t7BcBDLSeZAXKizEwEwvnbP3v93noS8g/view?usp=sharing"
    }
  ],
}
];

export default ROBOTICS_PROJECTS;