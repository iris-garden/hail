package is.hail;

import is.hail.QoBOutputStreamManager;
import java.io.*;
import java.lang.reflect.*;
import java.net.*;
import java.nio.*;
import java.nio.charset.*;
import java.util.*;
import java.util.concurrent.*;
import org.newsclub.net.unix.*;
import org.apache.logging.log4j.*;
import org.apache.logging.log4j.core.LoggerContext;
import org.apache.logging.log4j.core.config.Configurator;

enum ExitCode {
  USER_EXCEPTION,
  ENTRYWAY_EXCEPTION,
  NORMAL,
  CANCELLED,
  JVM_EOS, // NEVER USED ON JVM SIDE
  QOB_EXCEPTION
}

abstract class ClassLoaderRunnable implements Runnable {
  ClassLoader classLoader;
  ClassLoaderRunnable (ClassLoader classLoader) {
    this.classLoader = classLoader;
  }
  abstract void command();
  public void run() {
    ClassLoader oldClassLoader = Thread.currentThread().getContextClassLoader();
    Thread.currentThread().setContextClassLoader(this.classLoader);
    try {
      this.command();
    } finally {
      QoBOutputStreamManager.flushAllAppenders();
      Thread.currentThread().setContextClassLoader(oldClassLoader);
    }
  }
}

class JVMEntryway {
  private static final Logger log = LogManager.getLogger(JVMEntryway.class); // this will initialize log4j which is required for us to access the QoBOutputStreamManager in main
  static void finishException(ExitCode type, DataOutputStream out, Throwable t) throws IOException {
    out.writeInt(type.ordinal());
    try (StringWriter sw = new StringWriter();
         PrintWriter pw = new PrintWriter(sw)) {
      t.printStackTrace(pw);
      byte[] bytes = sw.toString().getBytes(StandardCharsets.UTF_8);
      out.writeInt(bytes.length);
      out.write(bytes);
    }
  }
  static Throwable cancelThreadRetrieveException(Future f) {
    f.cancel(true);
    return retrieveException(f);
  }
  static Throwable retrieveException(Future f) {
    try {
      f.get();
    } catch (CancellationException e) {
    } catch (Throwable t) {
      return t;
    }
    return null;
  }
  static void cancelAndAddSuppressed(Future f, Throwable t) {
    if (f != null) {
      Throwable t2 = cancelThreadRetrieveException(f);
      if (t2 != null) {
        t.addSuppressed(t2);
      }
    }
  }
  public static void main(String[] args) throws
    ClassNotFoundException, IllegalArgumentException, IOException, MalformedURLException,
    NoSuchMethodException, RuntimeException, SocketException, URISyntaxException {
    // parse arguments
    String socketAddress = args.length > 0 ? args[0] : null;
    if (socketAddress == null) {
      throw new IllegalArgumentException("missing positional argument 'socketAddress'");
    }

    // start server
    AFUNIXServerSocket server = AFUNIXServerSocket.newInstance();
    server.bind(new AFUNIXSocketAddress(new File(socketAddress)));
    System.err.println("listening on " + socketAddress);

    // test that reading/writing works on socket
    try (AFUNIXSocket socket = server.accept()) {
      System.err.println("negotiating start up with worker");
      DataInputStream in = new DataInputStream(socket.getInputStream());
      DataOutputStream out = new DataOutputStream(socket.getOutputStream());
      System.err.flush();
      out.writeBoolean(true);
      if (!in.readBoolean()) {
        System.err.println("read 'false' from socket");
        throw new SocketException("error writing to/reading from socket");
      }
    }

    // initialize variables outside of loop
    ExecutorService executor = Executors.newFixedThreadPool(2);
    HashMap<String, ClassLoader> classLoaders = new HashMap<>();

    // listen on socket
    while (true) {
      try (AFUNIXSocket socket = server.accept()) {
        System.err.println("connection accepted");
        DataInputStream in = new DataInputStream(socket.getInputStream());
        DataOutputStream out = new DataOutputStream(socket.getOutputStream());

        // parse arguments from socket
        int socketArgsLength = in.readInt();
        System.err.println("reading " + socketArgsLength + " arguments");
        String[] socketArgs = new String[socketArgsLength];
        for (int i = 0; i < socketArgsLength; ++i) {
          int length = in.readInt();
          byte[] bytes = new byte[length];
          System.err.println("reading " + i + ": length=" + length);
          in.read(bytes);
          socketArgs[i] = new String(bytes);
          System.err.println("reading " + i + ": " + socketArgs[i]);
        }
        if (socketArgs.length < 4) {
          throw new IllegalArgumentException("wrong number of arguments specified on socket (expected 4 or more)");
        }

        // get classloader from cache or create from url
        String classPath = socketArgs[0];
        ClassLoader cl = classLoaders.get(classPath);
        if (cl == null) {
          System.err.println("no extant classLoader for " + classPath);
          String[] urlStrings = classPath.split(",");
          ArrayList<URL> urls = new ArrayList<>();
          for (final String urlString : urlStrings) {
            File file = new File(urlString);
            urls.add(file.toURI().toURL());
            if (file.isDirectory()) {
              for (final File f : file.listFiles()) {
                urls.add(f.toURI().toURL());
              }
            }
          }
          cl = new URLClassLoader(urls.toArray(new URL[0]));
          classLoaders.put(classPath, cl);
        } else {
          System.err.println("reusing extant classLoader for " + classPath);
        }
        
        // load root class and get main method
        final ClassLoader hailRootCL = cl;
        Class<?> klass = hailRootCL.loadClass(socketArgs[1]);
        System.err.println("class loaded");
        Method main = klass.getDeclaredMethod("main", String[].class);
        System.err.println("main method loaded");

        QoBOutputStreamManager.changeFileInAllAppenders(socketArgs[3]);
        log.info("is.hail.JVMEntryway received arguments:");
        for (int i = 0; i < socketArgsLength; ++i) {
          log.info(i + ": " + socketArgs[i]);
        }
        log.info("Yielding control to the QoB Job.");

        // set up threads
        CompletionService<?> gather = new ExecutorCompletionService<Object>(executor);
        Future<?> mainThread = null;
        Future<?> shouldCancelThread = null;
        Future<?> completedThread = null;

        try {
          // submit threads
          mainThread = gather.submit(new ClassLoaderRunnable(hailRootCL) {
            void command() {
              try {
                main.invoke(null, (Object) Arrays.copyOfRange(socketArgs, 2, socketArgsLength));
              } catch (IllegalAccessException | InvocationTargetException e) {
                log.error("QoB Job threw an exception.", e);
                throw new RuntimeException(e);
              } catch (Exception e) {
                log.error("QoB Job threw an exception.", e);
              }
            }
          }, null);
          shouldCancelThread = gather.submit(new ClassLoaderRunnable(hailRootCL) {
            void command() {
              try {
                int i = in.readInt();
                if (i != 0) {
                  throw new RuntimeException(Integer.toString(i));
                }
              } catch (IOException e) {
                log.error("Exception encountered in QoB cancel thread.", e);
                throw new RuntimeException(e);
              } catch (Exception e) {
                log.error("Exception encountered in QoB cancel thread.", e);
              }
            }
          }, null);

          // wait for threads to complete and check which one completed
          completedThread = gather.take();
          if (completedThread == null) {
            throw new RuntimeException("no thread completed");
          }
          boolean wasCancelled = completedThread != mainThread;
          if (!wasCancelled) {
            System.err.println("main thread done");
          } else {
            if (completedThread != shouldCancelThread) {
              throw new RuntimeException("unknown thread completed");
            }
            System.err.println("cancelled");
          }

          // determine exit code and write output
          Throwable finishedException = retrieveException(wasCancelled ? shouldCancelThread : mainThread);
          Throwable secondaryException = cancelThreadRetrieveException(wasCancelled ? mainThread : shouldCancelThread);
          boolean hasFinishedException = finishedException != null;
          boolean hasSecondaryException = secondaryException != null;
          String finishedExceptionType = finishedException.getClass().getName();
          // FIXME is the exception class actually one of these when we get that exception from scala running in the thread?
          boolean hasHailException = finishedExceptionType == "HailBatchError" || finishedExceptionType == "HailWorkerFailure";
          if (hasFinishedException && hasSecondaryException) {
            finishedException.addSuppressed(secondaryException);
          }
          if (hasFinishedException && wasCancelled) {
            finishException(ExitCode.ENTRYWAY_EXCEPTION, out, finishedException);
          } else if (hasFinishedException && hasHailException) {
            finishException(ExitCode.QOB_EXCEPTION, out, finishedException);
          } else if (hasFinishedException) {
            finishException(ExitCode.USER_EXCEPTION, out, finishedException);
          } else if (hasSecondaryException && wasCancelled) {
            finishException(ExitCode.USER_EXCEPTION, out, secondaryException);
          } else if (hasSecondaryException) {
            finishException(ExitCode.ENTRYWAY_EXCEPTION, out, secondaryException);
          } else if (wasCancelled) {
            out.writeInt(ExitCode.CANCELLED.ordinal());
          } else {
            out.writeInt(ExitCode.NORMAL.ordinal());
          }
        } catch (Throwable entrywayException) {
          // cancel threads and propagate exceptions
          System.err.println("exception in entryway code");
          entrywayException.printStackTrace();
          cancelAndAddSuppressed(mainThread, entrywayException);
          cancelAndAddSuppressed(shouldCancelThread, entrywayException);
          finishException(ExitCode.ENTRYWAY_EXCEPTION, out, entrywayException);
        }
      } finally {
        QoBOutputStreamManager.flushAllAppenders();
        LoggerContext context = (LoggerContext) LogManager.getContext(false);
        ClassLoader loader = JVMEntryway.class.getClassLoader();
        URL url = loader.getResource("log4j2.properties");
        System.err.println("reconfiguring logging " + url.toString());
        context.setConfigLocation(url.toURI()); // this will force a reconfiguration
      }

      // restart loop and wait for another connection on socket
      System.err.println("waiting for next connection");
      System.err.flush();
      System.out.flush();
    }
  }
}
