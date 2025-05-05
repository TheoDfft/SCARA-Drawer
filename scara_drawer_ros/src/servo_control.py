#!/usr/bin/env python3
import sys
import time

# Try to import GPIO libraries with helpful error messages
try:
    import RPi.GPIO as GPIO
except ImportError:
    print("Error: RPi.GPIO library not found")
    print("Install it with: sudo apt install python3-rpi.gpio")
    print("You may also need: sudo apt install python3-pip")
    print("After installation, you might need to run with sudo: sudo python3 servo_control.py")
    sys.exit(1)

class ServoController:
    """Class to control a servo motor via PWM"""
    
    def __init__(self, pin=17, freq=50, initial_angle=0):
        """Initialize the servo controller
        
        Args:
            pin (int): GPIO pin number (BCM mode)
            freq (int): PWM frequency in Hz
            initial_angle (int): Initial angle to set the servo
        """
        # Define the GPIO pin for the servo
        self.servo_pin = pin
        
        # Configure GPIO settings
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        
        # Setup the GPIO pin for PWM
        GPIO.setup(self.servo_pin, GPIO.OUT)
        self.pwm = GPIO.PWM(self.servo_pin, freq)  # frequency in Hz (period = 1/freq)
        
        # Initialize servo
        self.pwm.start(0)
        
        # Keep track of current position
        self.current_position = None
        
        # Initialize servo position
        print(f"Initializing servo to {initial_angle}°")
        self.set_angle(initial_angle, smooth=False)
    
    def angle_to_duty_cycle(self, angle):
        """
        Convert angle (0-180) to duty cycle
        For pulse width of 500-2500 µsec on a 20ms period
        """
        # Map angle 0-180 to pulse width 500-2500 µsec
        pulse_width = 500 + (angle * (2000 / 180))
        # Convert pulse width (µsec) to duty cycle (%)
        duty_cycle = pulse_width / 20000 * 100
        return duty_cycle
    
    def _send_pulse(self, angle):
        """Send a single pulse to the servo at the specified angle"""
        duty = self.angle_to_duty_cycle(angle)
        self.pwm.ChangeDutyCycle(duty)
        
        # Convert angle to pulse width for display
        pulse_width = 500 + (angle * (2000 / 180))
        print(f"Setting angle: {angle:.1f}°, Pulse width: {int(pulse_width)} µsec, Duty cycle: {duty:.1f}%")
        
        # Allow time for servo to respond
        time.sleep(0.1)
        
        # Stop pulse to prevent jitter
        self.pwm.ChangeDutyCycle(0)
    
    def set_angle(self, target_angle, smooth=True, num_pulses=3):
        """Set servo to specified angle with improved reliability"""
        if not (0 <= target_angle <= 180):
            print("Error: Angle must be between 0 and 180 degrees")
            return False
        
        # Initialize current position if not set
        if self.current_position is None:
            self.current_position = 0
        
        # Determine if we should use smooth movement
        if smooth and self.current_position is not None:
            # Calculate number of steps based on angle difference
            angle_diff = abs(target_angle - self.current_position)
            num_steps = min(max(int(angle_diff / 10), 1), 10)  # Min 1, max 10 steps
            
            # Move in steps
            for step in range(1, num_steps + 1):
                intermediate_angle = self.current_position + (target_angle - self.current_position) * (step / num_steps)
                self._send_pulse(intermediate_angle)
                time.sleep(0.01)  # Short delay between steps
        
        # Send multiple pulses at the target angle for better accuracy
        for _ in range(num_pulses):
            self._send_pulse(target_angle)
            time.sleep(0.1)
        
        # Update current position
        self.current_position = target_angle
        return True
    
    def pen_up(self):
        """
        Lift the pen up by setting servo to 0 degrees
        """
        print("Lifting pen up")
        self.set_angle(0, smooth=True)
        
    def pen_down(self, bottom_angle=17):
        """
        Lower the pen down by setting servo to the specified bottom angle
        Default angle is 18 degrees
        """
        print(f"Lowering pen down to {bottom_angle} degrees")
        self.set_angle(bottom_angle, smooth=True)
    
    def cleanup(self):
        """Clean up GPIO resources"""
        self.pwm.stop()
        GPIO.cleanup()
        print("Cleanup complete")
    
    def run_interactive(self):
        """Run interactive command loop"""
        try:
            while True:
                # Get user input
                user_input = input("Enter an angle between 0 and 180, 'up', 'down', or 'q' to quit: ")
                
                # Check if user wants to quit
                if user_input.lower() == 'q':
                    break
                
                # Check for pen commands
                if user_input.lower() == 'up':
                    self.pen_up()
                    continue
                
                if user_input.lower() == 'down':
                    self.pen_down()
                    continue
                
                # Check if it's a custom pen down command with angle
                if user_input.lower().startswith('down '):
                    try:
                        angle_part = user_input.split(' ')[1]
                        bottom_angle = float(angle_part)
                        if 0 <= bottom_angle <= 180:
                            self.pen_down(bottom_angle)
                        else:
                            print("Error: Angle must be between 0 and 180 degrees")
                    except (ValueError, IndexError):
                        print("Error: Invalid format. Use 'down <angle>' where angle is a number")
                    continue
                
                # Try to convert input to a number
                try:
                    angle = float(user_input)
                    if 0 <= angle <= 180:
                        self.set_angle(angle, smooth=True)
                    else:
                        print("Error: Angle must be between 0 and 180 degrees")
                except ValueError:
                    print("Error: Please enter a valid number, 'up', 'down', or 'q' to quit")
        
        except KeyboardInterrupt:
            print("\nProgram stopped by user")
        
        finally:
            self.cleanup()

# Main execution
if __name__ == "__main__":
    # Create and run a servo controller
    servo = ServoController(pin=17, initial_angle=0)
    servo.run_interactive() 