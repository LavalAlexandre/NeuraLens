package com.example.ollamacameraapp

import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.ImageCapture
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.List
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.Text
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.lifecycle.viewmodel.compose.viewModel
import coil.compose.rememberAsyncImagePainter
import com.example.ollamacameraapp.data.ImageWithDescription
import com.example.ollamacameraapp.network.VertexAiClient
import com.example.ollamacameraapp.ui.components.CameraPreview
import com.example.ollamacameraapp.ui.components.CharacterIcon
import com.example.ollamacameraapp.ui.components.ImageDrawer
import com.example.ollamacameraapp.ui.components.ShutterButton
import com.example.ollamacameraapp.ui.components.SpeechBubble
import com.example.ollamacameraapp.ui.theme.OllamaCameraAppTheme
import com.example.ollamacameraapp.utils.takePictureAndSave
import com.example.ollamacameraapp.utils.takePictureForLLM
import java.io.File
import java.io.FileOutputStream
import java.util.Locale
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

@OptIn(ExperimentalMaterial3Api::class)
class MainActivity : ComponentActivity(), TextToSpeech.OnInitListener {

    private lateinit var tts: TextToSpeech

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        permissions.entries.forEach {
            Log.d("PermissionCallback", "${it.key} = ${it.value}")
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        tts = TextToSpeech(this, this)

        val permissionsToRequest = mutableListOf(Manifest.permission.CAMERA)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            permissionsToRequest.add(Manifest.permission.READ_MEDIA_IMAGES)
        } else {
            permissionsToRequest.add(Manifest.permission.READ_EXTERNAL_STORAGE)
        }

        val permissionsNotGranted = permissionsToRequest.filter {
            ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }

        if (permissionsNotGranted.isNotEmpty()) {
            requestPermissionLauncher.launch(permissionsNotGranted.toTypedArray())
        }

        setContent {
            OllamaCameraAppTheme {
                val viewModel: MainViewModel = viewModel()
                val images by viewModel.images.collectAsState()
                var showDrawer by remember { mutableStateOf(false) }
                val sheetState = rememberModalBottomSheetState()
                var selectedImage by remember { mutableStateOf<ImageWithDescription?>(null) }

                var description by remember { mutableStateOf("...") }
                var hasError by remember { mutableStateOf(false) }
                val client = remember { VertexAiClient() }
                val imageCapture = remember { ImageCapture.Builder().build() }
                val scope = rememberCoroutineScope()
                var showFlash by remember { mutableStateOf(false) }

                Box(modifier = Modifier.fillMaxSize()) {
                    CameraPreview(modifier = Modifier.fillMaxSize(), imageCapture = imageCapture)

                    if (showFlash) {
                        Box(
                            modifier = Modifier
                                .fillMaxSize()
                                .background(Color.White)
                        )
                    }

                    Column(modifier = Modifier.align(Alignment.BottomStart)) {
                        Box(contentAlignment = Alignment.BottomStart) {
                            // Always show character image
                            CharacterIcon()
                            // Only show speech bubble if not error and not initial state
                            if (!hasError && description != "...") {
                                SpeechBubble(
                                    text = description,
                                )
                            }
                        }

                        Box(
                            modifier = Modifier.fillMaxWidth().background(Color.Black).padding(16.dp),
                            contentAlignment = Alignment.BottomCenter,
                        ) {
                            ShutterButton(
                                onClick = {
                                    scope.launch {
                                        takePictureAndSave(imageCapture, this@MainActivity, description)
                                        showFlash = true
                                        delay(100) // flash duration
                                        showFlash = false
                                        viewModel.loadImages()
                                    }
                                }
                            )
                            IconButton(
                                onClick = {
                                    viewModel.loadImages()
                                    showDrawer = true
                                },
                                modifier = Modifier.align(Alignment.CenterStart).padding(start = 20.dp)
                            ) {
                                Icon(Icons.Default.List, contentDescription = "Open Gallery", tint = Color.White, modifier = Modifier.size(40.dp))
                            }
                        }
                    }

                }

                if (showDrawer) {
                    ModalBottomSheet(
                        onDismissRequest = { showDrawer = false },
                        sheetState = sheetState
                    ) {
                        ImageDrawer(images = images, onImageClick = {
                            selectedImage = it
                            scope.launch {
                                sheetState.hide()
                            }.invokeOnCompletion {
                                if (!sheetState.isVisible) {
                                    showDrawer = false
                                }
                            }
                        })
                    }
                }

                if (selectedImage != null) {
                    Box(modifier = Modifier.fillMaxSize().background(Color.Black)) {
                        Image(
                            painter = rememberAsyncImagePainter(selectedImage!!.uri),
                            contentDescription = "Full screen image",
                            modifier = Modifier.fillMaxSize(),
                            contentScale = ContentScale.Fit
                        )
                        selectedImage!!.description?.let {
                            Text(
                                text = it,
                                color = Color.White,
                                modifier = Modifier.align(Alignment.BottomCenter).padding(16.dp)
                            )
                        }
                        IconButton(
                            onClick = {
                                selectedImage = null
                                showDrawer = true
                             },
                            modifier = Modifier.align(Alignment.TopEnd).padding(16.dp)
                        ) {
                            Icon(Icons.Default.Close, contentDescription = "Close", tint = Color.White)
                        }
                    }
                }

                LaunchedEffect(Unit) {
                    while (true) {
                        try {
                            val file = takePictureForLLM(imageCapture, this@MainActivity)
                            description = client.getDescription(file, BuildConfig.VERTEX_ACCESS_TOKEN)
                            hasError = description.startsWith("Error:")

                            if (!hasError && description != "...") {
                                tts.speak(description, TextToSpeech.QUEUE_FLUSH, null, "")
                            }

                            file.delete()
                        } catch (e: Exception) {
                            description = e.message ?: "Error"
                            hasError = true
                        }
                        delay(5000)
                    }
                }
            }
        }
    }

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            val result = tts.setLanguage(Locale.US)
            if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                Log.e("TTS", "The Language specified is not supported!")
            }
        } else {
            Log.e("TTS", "Initialization Failed!")
        }
    }

    override fun onDestroy() {
        if (::tts.isInitialized) {
            tts.stop()
            tts.shutdown()
        }
        super.onDestroy()
    }
}
