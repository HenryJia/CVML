##
## Auto Generated makefile by CodeLite IDE
## any manual changes will be erased      
##
## Debug
ProjectName            :=CVML
ConfigurationName      :=Debug
WorkspacePath          := "/home/henry/Coding/C++"
ProjectPath            := "/home/henry/Coding/C++/CVML"
IntermediateDirectory  :=./Debug
OutDir                 := $(IntermediateDirectory)
CurrentFileName        :=
CurrentFilePath        :=
CurrentFileFullPath    :=
User                   :=
Date                   :=29/05/15
CodeLitePath           :="/home/henry/.codelite"
LinkerName             :=/usr/bin/g++
SharedObjectLinkerName :=/usr/bin/g++ -shared -fPIC
ObjectSuffix           :=.o
DependSuffix           :=.o.d
PreprocessSuffix       :=.i
DebugSwitch            :=-g 
IncludeSwitch          :=-I
LibrarySwitch          :=-l
OutputSwitch           :=-o 
LibraryPathSwitch      :=-L
PreprocessorSwitch     :=-D
SourceSwitch           :=-c 
OutputFile             :=$(IntermediateDirectory)/$(ProjectName)
Preprocessors          :=
ObjectSwitch           :=-o 
ArchiveOutputSwitch    := 
PreprocessOnlySwitch   :=-E
ObjectsFileList        :="CVML.txt"
PCHCompileFlags        :=
MakeDirCommand         :=mkdir -p
LinkOptions            :=  
IncludePath            :=  $(IncludeSwitch). $(IncludeSwitch). 
IncludePCH             := 
RcIncludePath          := 
Libs                   := $(LibrarySwitch)opencv_core $(LibrarySwitch)opencv_features2d $(LibrarySwitch)opencv_flann $(LibrarySwitch)opencv_gpu $(LibrarySwitch)opencv_highgui $(LibrarySwitch)opencv_imgproc $(LibrarySwitch)opencv_legacy $(LibrarySwitch)opencv_ml $(LibrarySwitch)opencv_nonfree $(LibrarySwitch)opencv_objdetect $(LibrarySwitch)opencv_ocl $(LibrarySwitch)opencv_photo $(LibrarySwitch)opencv_stitching $(LibrarySwitch)opencv_superres $(LibrarySwitch)opencv_ts $(LibrarySwitch)opencv_video $(LibrarySwitch)opencv_videostab $(LibrarySwitch)tbb $(LibrarySwitch)rt $(LibrarySwitch)pthread $(LibrarySwitch)m $(LibrarySwitch)dl 
ArLibs                 :=  "opencv_core" "opencv_features2d" "opencv_flann" "opencv_gpu" "opencv_highgui" "opencv_imgproc" "opencv_legacy" "opencv_ml" "opencv_nonfree" "opencv_objdetect" "opencv_ocl" "opencv_photo" "opencv_stitching" "opencv_superres" "opencv_ts" "opencv_video" "opencv_videostab" "tbb" "rt" "pthread" "m" "dl" 
LibPath                := $(LibraryPathSwitch). 

##
## Common variables
## AR, CXX, CC, AS, CXXFLAGS and CFLAGS can be overriden using an environment variables
##
AR       := /usr/bin/ar rcu
CXX      := /usr/bin/g++
CC       := /usr/bin/gcc
CXXFLAGS :=  -g -O1 -O -O3 -O0 -O2 -std=c++11 -Wall -I /usr/include/opencv/ $(Preprocessors)
CFLAGS   :=  -g -O1 -O -O3 -O0 -O2 -std=c++11 -Wall -I /usr/include/opencv/ $(Preprocessors)
ASFLAGS  := 
AS       := /usr/bin/as


##
## User defined environment variables
##
CodeLiteDir:=/usr/share/codelite
Objects0=$(IntermediateDirectory)/main.cpp$(ObjectSuffix) $(IntermediateDirectory)/cvnn.cpp$(ObjectSuffix) 



Objects=$(Objects0) 

##
## Main Build Targets 
##
.PHONY: all clean PreBuild PrePreBuild PostBuild
all: $(OutputFile)

$(OutputFile): $(IntermediateDirectory)/.d $(Objects) 
	@$(MakeDirCommand) $(@D)
	@echo "" > $(IntermediateDirectory)/.d
	@echo $(Objects0)  > $(ObjectsFileList)
	$(LinkerName) $(OutputSwitch)$(OutputFile) @$(ObjectsFileList) $(LibPath) $(Libs) $(LinkOptions)

$(IntermediateDirectory)/.d:
	@test -d ./Debug || $(MakeDirCommand) ./Debug

PreBuild:


##
## Objects
##
$(IntermediateDirectory)/main.cpp$(ObjectSuffix): main.cpp $(IntermediateDirectory)/main.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/henry/Coding/C++/CVML/main.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/main.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/main.cpp$(DependSuffix): main.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/main.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/main.cpp$(DependSuffix) -MM "main.cpp"

$(IntermediateDirectory)/main.cpp$(PreprocessSuffix): main.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/main.cpp$(PreprocessSuffix) "main.cpp"

$(IntermediateDirectory)/cvnn.cpp$(ObjectSuffix): cvnn.cpp $(IntermediateDirectory)/cvnn.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/henry/Coding/C++/CVML/cvnn.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/cvnn.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/cvnn.cpp$(DependSuffix): cvnn.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/cvnn.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/cvnn.cpp$(DependSuffix) -MM "cvnn.cpp"

$(IntermediateDirectory)/cvnn.cpp$(PreprocessSuffix): cvnn.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/cvnn.cpp$(PreprocessSuffix) "cvnn.cpp"


-include $(IntermediateDirectory)/*$(DependSuffix)
##
## Clean
##
clean:
	$(RM) -r ./Debug/


