import './App.css';
import { ChakraProvider, Box, VStack, HStack, Heading, Text, Button, Image, Icon, Tabs, TabList, TabPanels, Tab, TabPanel } from '@chakra-ui/react';
import { FaGithub } from 'react-icons/fa'; // Importing the GitHub icon from Font Awesome
import Bibtex from './Bibtex';
import Authors from './Authors';

function App() {
  return (
    <div className="paper-container">
    <ChakraProvider>
      <Box maxWidth="1200px" margin="auto" padding="4">
        <VStack spacing={8} align="stretch">
          <Heading as="h1" size="2xl" textAlign="center">
          SD-πXL: Generating Low-Resolution Quantized Imagery via Score Distillation
          </Heading>
          
          <Authors/>

          <Text mt={2} textAlign="center" fontSize="xl">
            <Button as="a" href="https://asia.siggraph.org/2024/" colorScheme="teal" variant="solid">
              SIGGRAPH Asia '24
            </Button>
          </Text>
          
          <HStack spacing={4} justify="center">
            <Button as="a" href="/SD-piXL/paper/sd-pixl.pdf" colorScheme="blue">
              Paper (30MB)
            </Button>
            <Button as="a" href="/SD-piXL/paper/sd-pixl_supplementary_material.pdf" colorScheme="blue">
              Supplementary (60MB)
            </Button>
            <Button as="a" href="https://github.com/AlexandreBinninger/SD-piXL" colorScheme="green" leftIcon={<Icon as={FaGithub} />}>
              Source Code
            </Button>
            <Button as="a" href="https://doi.org/10.1145/3680528.3687570" colorScheme="red">
              ACM 
            </Button>
          </HStack>
          
          <Box textAlign="center" mt={4}>
            <Image src="/SD-piXL/paper/teaser.jpg" alt="Teaser Image" width="100%" />
            <Text fontSize="sm" color="gray.600" mt={2}>
            SD-πXL specializes in creating pixel art, characterized by its intentionally low resolution and limited color palette. Our method enables varying degrees of control: the input is a text prompt, and optionally a reference (high-resolution) image for initialization or spatial control. SD-πXL's output style can be adjusted using fine-tuned diffusion models. In this example, the full prompt reads ''Embroidery of a Chinese dragon flying through the air on a dark background with smoke coming out of its mouth and tail.''. The output pixel art can be used for crafted fabrications, such as the shown cross-stitch embroidery.
            </Text>
          </Box>

          
          <Box>
            <Heading as="h2" size="xl">Abstract</Heading>
            <Text mt={2}>Low-resolution quantized imagery, such as pixel art, is seeing a revival in modern applications ranging from video game graphics to digital design and fabrication, where creativity is often bound by a limited palette of elemental units. Despite their growing popularity, the automated generation of quantized images from raw inputs remains a significant challenge, often necessitating intensive manual input. We introduce SD-πXL, an approach for producing quantized images that employs score distillation sampling in conjunction with a differentiable image generator. Our method enables users to input a prompt and optionally an image for spatial conditioning, set any desired output size H &times; W, and choose a palette of n colors or elements. Each color corresponds to a distinct class for our generator, which operates on an H &times; W &times; n tensor. We adopt a softmax approach, computing a convex sum of elements, thus rendering the process differentiable and amenable to backpropagation. We show that employing Gumbel-softmax reparameterization allows for crisp pixel art effects. Unique to our method is the ability to transform input images into low-resolution, quantized versions while retaining their key semantic features. Our experiments validate SD-πXL's performance in creating visually pleasing and faithful representations, consistently outperforming the current state-of-the-art. Furthermore, we showcase SD-πXL's practical utility in fabrication through its applications in interlocking brick mosaic, beading and embroidery design.</Text>
          </Box>
          
          {/* <Box>
            <Heading as="h2" size="xl">Results</Heading>
            <Text mt={2}>Description of results...</Text>
            <Image src="path-to-results-image.jpg" alt="Results" mt={4} width="100%" />
          </Box> */}
          <Box>
      <Heading as="h2" size="xl">Results</Heading>
      <Text mt={2}>Explore the different results:</Text>

      <Tabs isFitted variant="enclosed">
        <TabList mb="1em">
          <Tab>Result 1</Tab>
          <Tab>Result 2</Tab>
          <Tab>Result 3</Tab>
        </TabList>
        <TabPanels>
          <TabPanel>
            <Image src="path-to-result-1-image.jpg" alt="Result 1" width="100%" />
            <Text fontSize="sm" mt={2}>Description of Result 1</Text>
          </TabPanel>
          <TabPanel>
            <Image src="path-to-result-2-image.jpg" alt="Result 2" width="100%" />
            <Text fontSize="sm" mt={2}>Description of Result 2</Text>
          </TabPanel>
          <TabPanel>
            <Image src="path-to-result-3-image.jpg" alt="Result 3" width="100%" />
            <Text fontSize="sm" mt={2}>Description of Result 3</Text>
          </TabPanel>
        </TabPanels>
      </Tabs>
    </Box>

          <Box>
            <Heading as="h2" size="xl">Citation</Heading>
            <Bibtex path={'/SD-piXL/paper/cite.bib'}/>
          </Box>

          <Box>
            <Heading as="h2" size="xl">Acknowledgments</Heading>
            <Text mt={2}>We thank the anonymous reviewers for their constructive feedback and Danielle Luterbacher for her help with setting up the embroidery machine. Ximing Xing's open-source version of VectorFusion was instrumental in the development and design of our source code. This work was supported in part by the European Research Council (ERC) under the European Union's Horizon 2020 research and innovation program (grant agreement No. 101003104, ERC CoG MYCLOTH).</Text>
          </Box>
        </VStack>
      </Box>
    </ChakraProvider>
  </div>
  );
}

export default App;
