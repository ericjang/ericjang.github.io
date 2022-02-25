
passive walker 
http://ruina.tam.cornell.edu/research/topics/locomotion_and_robotics/3d_passive_dynamic/3d_passive_dynamic.pdf


https://arxiv.org/pdf/2103.04675.pdf

http://www.cs.columbia.edu/~allen/S19/

https://twitter.com/techatfacebook/status/1460679048099549185?s=20


FB reality labs
https://tech.fb.com/inside-reality-labs-meet-the-team-thats-bringing-touch-to-the-digital-world/


https://www.youtube.com/watch?v=CwdQjO5_OhI&t=39s



Keywords: JSK lab, Schaft, Ishiguro Robot, Honda Asimo, Telesar from Tachi lab, 
@telexistenceinc
, Softbank Pepper, Robo-One, Gundam cafe, Doraemon.


https://www.mkoval.org/projects/graduate/darpa-arm-s

Teleoperation:


disney's hydrostatic transmissions - near instantaneous feedback

https://www.youtube.com/watch?v=HY4bfnHMdtk

https://s3-us-west-1.amazonaws.com/disneyresearch/wp-content/uploads/20160503162533/A-Hybrid-Hydrostatic-Transmission-and-Human-Safe-Haptic-Telepresence-Robot-Paper.pdf

https://www.youtube.com/watch?v=1ehI2cKi4v0

simulators:
https://arxiv.org/pdf/2008.04627.pdf



halodi
instead of high-ratio gearboxes, use high torque with low transmission ratios. increase motor's inertia and reduce gear ratio - lower motor inertia reflected to end-effector.

low ratios have less friction and backlash

downside: moderate gear-ratio cannot compensate non-linear coupling terms like cogging torque


direct drive motor - reduces need for coupling/transmission (which make noise/ add parts)

high torque, light weight, compact

frameless - no gearbox surrounding the motor

https://www.youtube.com/watch?v=6nPlctPZwv4



brushless - longer lived

permanet magnet and electromagnet
rotor (spinning) and stator


quadruped https://arxiv.org/pdf/1910.00093.pdf


trifinger: hands - based on the quadruped
https://arxiv.org/pdf/2008.03596.pdf


low gear ratio - suitable for impedance and force control



BLDC - brushless DC motor

harmonic drive (strain wave gear)
https://www.youtube.com/watch?v=xlnNj9F37MA

high gear ratio in a small form factor - allows you to convert speed ->  torque.

harmonic drives driven by either BLDC or Stepper motor.


Harmonic drives were once popular for robot arms, because they provide huge gear reductions in a small space. But they're not back-drivable at all - overload one and you break gear teeth. As motor control has improved, the need for huge gear reductions has decreased, and harmonic drives have fallen out of favor in industrial robots.

passive compliance - it just moves
active compliance - senses output torques and performs feedback control (series elastic actuation )

Kuka LBR - closed-loop strained based force control, 67k 
Franka - similar, 30k
sawyer 30k

all of above use harmonic drives, which can be made backdriveable with additional sensors but not inherently compliant

rethink baxter 25k - series elastic robots (SEA actuator)

QDD = high torque motor + low-ratio transmission (<24:1)

backdriveability is inversely proportional to square of transmission ratio. so high transmission ratio means very not backdriveable.

low backdrive torque, high output torque
- free-swinging knee motion
- powerful push
- compliance to impacts
- high efficiency and regenerative breaking
- low acoustic noise

meshing of parts contributes to noise.



Backdriveable:

Barrett WAM - 135k - backdriveable. all actuators in base, low-friction cable transmissions
PR2 - 400k - backdriveability thru gravity compensation


direct drive is most backdriveable, but high motor mass in arm makes high-DoF impractical

Quasi-Direct Drive (QDD transmission ratio < 1:10) used for legged locomition - have low friction, high backdriveability, toughness, simplicity, force control, selectable impedance. Main drawback is reduced torque density.

high torque removes need for high gear ratio


The legs
https://www.youtube.com/watch?v=6RvMVpmYSV8


commercially driven exoskeletons are too stiff , backdriveability is key

https://hackaday.io/project/160882-blackbird-bipedal-robot



The arms

Backdriveable motors

https://hackaday.io/project/159404-opentorque-actuator

Hebi robotics - series actuator 
https://www.hebirobotics.com/hardware

human has about 10hz control bandwidth


hardware cost is dominated by electric motors and their transmissions.


japanese robots
JSK lab, Schaft, Ishiguro Robot, 

Honda Asimo

Telesar from Tachi lab
https://tachilab.org/en/projects/telesar.html


@telexistenceinc https://t.co/EfG1YMQ2eK
, Softbank Pepper, Robo-One, Gundam cafe, Doraemon

TALOS

https://hal.archives-ouvertes.fr/hal-01485519/file/ichr-talos.pdf


compact gearboxes for modern robotics
https://www.frontiersin.org/articles/10.3389/frobt.2020.00103/pdf
