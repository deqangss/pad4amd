import java.util.Random;

public class Opaque {
  static void opaque() {
    Random random = new Random();

    boolean variables[] = new boolean[40];
    for (int i = 0; i < variables.length; i++) {
      variables[i] = random.nextBoolean();
    }

    boolean constant = true;
    for (int i = 0; i < 40 * 4.6; i++) {
      boolean clause = false;

      for (int j = 0; j < 3; j++) {
        int choice = random.nextInt(variables.length);
        clause |= variables[choice];
      }
      if (!clause) {
        constant = false;
      }

    }
    if (constant) {
      System.out.println("opaque");  // This gets replaced with injected gadget
    }
  }
}