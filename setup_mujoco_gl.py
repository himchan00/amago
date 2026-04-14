"""Setup GL headers and libraries for mujoco_py JIT compilation on headless clusters.

mujoco_py 2.1.2.14 selects the CPU/OSMesa builder when nvidia-smi or nvidia lib
dirs are not found. This script provides:
  1) GL/OSMesa headers from Mesa gitlab
  2) Stub libOSMesa.so (MetaWorld doesn't render, so stubs suffice)
  3) Real system libGL.so via symlink, or comprehensive stub as fallback
"""
import glob
import os
import subprocess
import tempfile
import urllib.request

CONDA_ENV = "/home/aiscuser/.conda/envs/amago"
INC_DIR = os.path.join(CONDA_ENV, "include")
LIB_DIR = os.path.join(CONDA_ENV, "lib")

# ---------------------------------------------------------------------------
# 1) Download GL headers (osmesa.h -> gl.h -> glext.h, khrplatform.h)
# ---------------------------------------------------------------------------
MESA_BASE = "https://gitlab.freedesktop.org/mesa/mesa/-/raw/mesa-22.2.5/include"
HEADERS = ["GL/osmesa.h", "GL/gl.h", "GL/glext.h", "KHR/khrplatform.h"]

for h in HEADERS:
    dest = os.path.join(INC_DIR, h)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    urllib.request.urlretrieve(f"{MESA_BASE}/{h}", dest)
print("GL headers downloaded OK")

# ---------------------------------------------------------------------------
# 2) Compile stub libOSMesa.so (full OSMesa API)
# ---------------------------------------------------------------------------
OSMESA_STUB = """\
typedef void* OSMesaContext;
typedef int GLint; typedef unsigned int GLsizei;
typedef unsigned int GLenum; typedef unsigned char GLboolean;
OSMesaContext OSMesaCreateContext(GLenum f,OSMesaContext s){return 0;}
OSMesaContext OSMesaCreateContextExt(GLenum f,GLint d,GLint s,GLint a,OSMesaContext sh){return 0;}
OSMesaContext OSMesaCreateContextAttribs(const int*a,OSMesaContext sh){return 0;}
void OSMesaDestroyContext(OSMesaContext c){}
GLboolean OSMesaMakeCurrent(OSMesaContext c,void*b,GLenum t,GLsizei w,GLsizei h){return 0;}
OSMesaContext OSMesaGetCurrentContext(void){return 0;}
void OSMesaPixelStore(GLint p,GLint v){}
void OSMesaGetIntegerv(GLint p,GLint*v){if(v)*v=0;}
GLboolean OSMesaGetDepthBuffer(OSMesaContext c,GLint*w,GLint*h,GLint*b,void**p){return 0;}
GLboolean OSMesaGetColorBuffer(OSMesaContext c,GLint*w,GLint*h,GLint*f,void**p){return 0;}
void* OSMesaGetProcAddress(const char*f){return 0;}
void OSMesaColorClamp(GLboolean e){}
"""
f = tempfile.NamedTemporaryFile(suffix=".c", mode="w", delete=False)
f.write(OSMESA_STUB)
f.close()
subprocess.check_call(["gcc", "-shared", "-o", f"{LIB_DIR}/libOSMesa.so", f.name])
os.unlink(f.name)
print("Stub libOSMesa.so OK")

# ---------------------------------------------------------------------------
# 3) Find real system libGL.so or compile comprehensive stub
# ---------------------------------------------------------------------------
GL_TARGET = os.path.join(LIB_DIR, "libGL.so")
if os.path.exists(GL_TARGET) or os.path.islink(GL_TARGET):
    os.remove(GL_TARGET)

# Try system paths first
found = False
for d in ["/usr/lib/x86_64-linux-gnu", "/usr/lib64", "/usr/lib", "/usr/local/nvidia/lib64"]:
    for c in sorted(glob.glob(os.path.join(d, "libGL.so*"))):
        if os.path.isfile(c):
            os.symlink(c, GL_TARGET)
            print(f"libGL.so -> {c}")
            found = True
            break
    if found:
        break

if not found:
    # Try ldconfig
    try:
        r = subprocess.run(["ldconfig", "-p"], capture_output=True, text=True, timeout=10)
        for line in r.stdout.split("\n"):
            if "libGL.so" in line and "=>" in line:
                path = line.split("=>")[-1].strip()
                if os.path.isfile(path):
                    os.symlink(path, GL_TARGET)
                    print(f"libGL.so -> {path} (via ldconfig)")
                    found = True
                    break
    except Exception:
        pass

if not found:
    print("No system libGL found. Compiling comprehensive stub...")
    GL_STUB = """\
typedef unsigned int GLenum;typedef int GLint;typedef unsigned int GLuint;
typedef int GLsizei;typedef float GLfloat;typedef double GLdouble;
typedef unsigned char GLubyte;typedef unsigned char GLboolean;
typedef float GLclampf;typedef int GLbitfield;
void glBegin(GLenum m){}void glEnd(void){}
void glNormal3f(GLfloat x,GLfloat y,GLfloat z){}void glNormal3fv(const GLfloat*v){}
void glVertex3f(GLfloat x,GLfloat y,GLfloat z){}void glVertex3fv(const GLfloat*v){}
void glVertex2f(GLfloat x,GLfloat y){}
void glColor3f(GLfloat r,GLfloat g,GLfloat b){}void glColor4f(GLfloat r,GLfloat g,GLfloat b,GLfloat a){}
void glColor3fv(const GLfloat*v){}void glColor4fv(const GLfloat*v){}void glTexCoord2f(GLfloat s,GLfloat t){}
void glClear(GLbitfield m){}void glClearColor(GLclampf r,GLclampf g,GLclampf b,GLclampf a){}
void glClearDepth(GLdouble d){}void glViewport(GLint x,GLint y,GLsizei w,GLsizei h){}
void glScissor(GLint x,GLint y,GLsizei w,GLsizei h){}
void glMatrixMode(GLenum m){}void glLoadIdentity(void){}
void glMultMatrixf(const GLfloat*m){}void glMultMatrixd(const GLdouble*m){}
void glLoadMatrixf(const GLfloat*m){}void glLoadMatrixd(const GLdouble*m){}
void glPushMatrix(void){}void glPopMatrix(void){}
void glOrtho(GLdouble l,GLdouble r,GLdouble b,GLdouble t,GLdouble n,GLdouble f){}
void glFrustum(GLdouble l,GLdouble r,GLdouble b,GLdouble t,GLdouble n,GLdouble f){}
void glTranslatef(GLfloat x,GLfloat y,GLfloat z){}void glRotatef(GLfloat a,GLfloat x,GLfloat y,GLfloat z){}
void glScalef(GLfloat x,GLfloat y,GLfloat z){}
void glEnable(GLenum c){}void glDisable(GLenum c){}GLboolean glIsEnabled(GLenum c){return 0;}
void glEnableClientState(GLenum c){}void glDisableClientState(GLenum c){}
void glGetIntegerv(GLenum p,GLint*v){if(v)*v=0;}void glGetFloatv(GLenum p,GLfloat*v){if(v)*v=0;}
void glGetDoublev(GLenum p,GLdouble*v){if(v)*v=0;}
const GLubyte*glGetString(GLenum n){return(const GLubyte*)"";}GLenum glGetError(void){return 0;}
void glLightfv(GLenum l,GLenum p,const GLfloat*v){}void glLightf(GLenum l,GLenum p,GLfloat v){}
void glLightModelfv(GLenum p,const GLfloat*v){}void glLightModeli(GLenum p,GLint v){}
void glMaterialfv(GLenum f,GLenum p,const GLfloat*v){}void glMaterialf(GLenum f,GLenum p,GLfloat v){}
void glShadeModel(GLenum m){}void glDepthFunc(GLenum f){}void glDepthMask(GLboolean f){}
void glDepthRange(GLdouble n,GLdouble f){}void glAlphaFunc(GLenum f,GLclampf r){}
void glBlendFunc(GLenum s,GLenum d){}void glStencilFunc(GLenum f,GLint r,GLuint m){}
void glStencilOp(GLenum f,GLenum z,GLenum p){}
void glCullFace(GLenum m){}void glFrontFace(GLenum m){}void glPolygonMode(GLenum f,GLenum m){}
void glPolygonOffset(GLfloat f,GLfloat u){}void glLineWidth(GLfloat w){}void glPointSize(GLfloat s){}
void glTexParameteri(GLenum t,GLenum p,GLint v){}void glTexParameterf(GLenum t,GLenum p,GLfloat v){}
void glTexParameterfv(GLenum t,GLenum p,const GLfloat*v){}
void glGenTextures(GLsizei n,GLuint*t){int i;if(t)for(i=0;i<n;i++)t[i]=i+1;}
void glDeleteTextures(GLsizei n,const GLuint*t){}void glBindTexture(GLenum t,GLuint x){}
void glTexImage2D(GLenum t,GLint l,GLint i,GLsizei w,GLsizei h,GLint b,GLenum f,GLenum p,const void*d){}
void glTexSubImage2D(GLenum t,GLint l,GLint x,GLint y,GLsizei w,GLsizei h,GLenum f,GLenum p,const void*d){}
void glPixelStorei(GLenum p,GLint v){}
void glReadPixels(GLint x,GLint y,GLsizei w,GLsizei h,GLenum f,GLenum t,void*d){}
void glReadBuffer(GLenum m){}void glDrawBuffer(GLenum m){}void glFlush(void){}void glFinish(void){}
void glHint(GLenum t,GLenum m){}void glFogf(GLenum p,GLfloat v){}void glFogfv(GLenum p,const GLfloat*v){}
void glFogi(GLenum p,GLint v){}void glColorMask(GLboolean r,GLboolean g,GLboolean b,GLboolean a){}
void glVertexPointer(GLint s,GLenum t,GLsizei st,const void*p){}
void glNormalPointer(GLenum t,GLsizei st,const void*p){}
void glColorPointer(GLint s,GLenum t,GLsizei st,const void*p){}
void glTexCoordPointer(GLint s,GLenum t,GLsizei st,const void*p){}
void glDrawArrays(GLenum m,GLint f,GLsizei c){}void glDrawElements(GLenum m,GLsizei c,GLenum t,const void*i){}
void glGenFramebuffers(GLsizei n,GLuint*f){int i;if(f)for(i=0;i<n;i++)f[i]=i+1;}
void glDeleteFramebuffers(GLsizei n,const GLuint*f){}void glBindFramebuffer(GLenum t,GLuint f){}
void glGenRenderbuffers(GLsizei n,GLuint*r){int i;if(r)for(i=0;i<n;i++)r[i]=i+1;}
void glDeleteRenderbuffers(GLsizei n,const GLuint*r){}void glBindRenderbuffer(GLenum t,GLuint r){}
void glRenderbufferStorage(GLenum t,GLenum f,GLsizei w,GLsizei h){}
void glFramebufferRenderbuffer(GLenum t,GLenum a,GLenum rt,GLuint r){}
void glFramebufferTexture2D(GLenum t,GLenum a,GLenum tt,GLuint tex,GLint l){}
GLenum glCheckFramebufferStatus(GLenum t){return 0x8CD5;}
void*glXGetCurrentDisplay(void){return 0;}
"""
    f = tempfile.NamedTemporaryFile(suffix=".c", mode="w", delete=False)
    f.write(GL_STUB)
    f.close()
    subprocess.check_call(["gcc", "-shared", "-o", GL_TARGET, f.name])
    os.unlink(f.name)
    print("Comprehensive stub libGL.so compiled OK")

print(f"libGL.so ready: {os.path.realpath(GL_TARGET)}")
