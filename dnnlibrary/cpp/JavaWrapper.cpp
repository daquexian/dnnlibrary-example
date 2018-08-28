//
// Created by daquexian on 2017/11/12.
//

#include <android/asset_manager_jni.h>
#include <map>
#include <common/helper.h>
#include <ModelBuilder.h>
#include <DaqReader.h>

using std::string; using std::map;

ModelBuilder builder;
DaqReader daq_reader;

std::unique_ptr<Model> model;

bool isUsing;

jint throwException(JNIEnv *env, std::string message);


extern "C"
JNIEXPORT void
JNICALL
Java_me_daquexian_dnnlibrary_ModelWrapper_readFile(
        JNIEnv *env,
        jobject /* this */,
        jobject javaAssetManager, jstring javaFilename) {

    if (isUsing) {
        throwException(env, "Please compile current model before generate a new model");
    }

    isUsing = true;

    string filename = string(env->GetStringUTFChars(javaFilename, nullptr));
    AAssetManager *mgrr = AAssetManager_fromJava(env, javaAssetManager);

    AAsset* asset = AAssetManager_open(mgrr, filename.c_str(), AASSET_MODE_UNKNOWN);
    off_t start, length;
    auto fd = AAsset_openFileDescriptor(asset, &start, &length);

    daq_reader.ReadDaq(fd, builder);
}

extern "C"
JNIEXPORT void
JNICALL
Java_me_daquexian_dnnlibrary_ModelWrapper_setOutput(
        JNIEnv *env,
        jobject /* this */,
        jstring javaBlobName) {
    if (!isUsing) {
        throwException(env, "No model to add output");
    }
    string blobName = string(env->GetStringUTFChars(javaBlobName, nullptr));
    model->addIndexIntoOutput(builder.getBlobIndex(blobName));
}

extern "C"
JNIEXPORT void
JNICALL
Java_me_daquexian_dnnlibrary_ModelWrapper_compile(
        JNIEnv *env,
        jobject /* this */,
        jint preference) {
    if (!isUsing) {
        throwException(env, "No model to compile");
    }

    builder.compile(preference);
    model = builder.finish();

    isUsing = false;
}

extern "C"
JNIEXPORT jfloatArray
JNICALL
Java_me_daquexian_dnnlibrary_ModelWrapper_predict(
        JNIEnv *env,
        jobject /* this */,
        jfloatArray dataArrayObject) {

    jfloat *data = env->GetFloatArrayElements(dataArrayObject, nullptr);
    jsize dataLen = env->GetArrayLength(dataArrayObject) * sizeof(jfloat);

    model->setInputBuffer(builder.getInputIndexes()[0], data, static_cast<size_t>(dataLen));

    uint32_t outputLen = product(builder.getBlobDim(builder.getOutputIndexes()[0]));
    float output[outputLen];
    model->setOutputBuffer(builder.getOutputIndexes()[0], output, sizeof(output));

    model->predict();

    jfloatArray result = env->NewFloatArray(product(builder.getBlobDim(builder.getOutputIndexes()[0])));
    env->SetFloatArrayRegion(result, 0, outputLen, output);

    return result;
}

jint throwException(JNIEnv *env, std::string message) {
    jclass exClass;
    std::string className = "java/lang/RuntimeException" ;

    exClass = env->FindClass(className.c_str());

    return env->ThrowNew(exClass, message.c_str());
}

